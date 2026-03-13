"""Encapsulate push-model evaluation and rate-limiting.

This class manages a small in-memory history for incoming MQTT data,
evaluates a user-configured predicate (a safe expression) and calls a
publisher when the predicate transitions true->false according to
cooldown and hourly limits.

Design notes:
- This is intentionally small and depends on `PushHistoryCache` for
  history storage. More sophisticated workflow orchestration should be
  implemented in `indi_allsky/workflows.py` (DataPackager/AnalysisWorkflow).
"""

import time
import logging
import ast
import json
from datetime import datetime
from typing import Callable

from .push_history import PushHistoryCache
from .rain_detection_models import trend_slope as util_trend_slope, increasing as util_increasing, decreasing as util_decreasing
from .workflows import DataPackager, AnalysisWorkflow, PushMessaging, TimeGate
from . import rain_detection_models as rdm

logger = logging.getLogger('indi_allsky')


class UnsafeExpression(Exception):
    pass


def _safe_eval(expr: str, ctx: dict, allowed_calls: set):
    """Evaluate a user expression with basic safety checks.

    Only a very small subset of expressions is allowed; function calls
    are restricted to names in `allowed_calls`.
    """
    node = ast.parse(expr, mode='eval')
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.Delete, ast.Lambda, ast.Global, ast.With, ast.Try)):
            raise UnsafeExpression('Forbidden AST node')
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name):
                if func.id not in allowed_calls:
                    raise UnsafeExpression('Call to disallowed function')
            else:
                raise UnsafeExpression('Only simple function names are allowed')
    return eval(compile(node, '<push_model>', 'eval'), {'__builtins__': None}, ctx)


class PushEvaluator:
    def __init__(self, config: dict, publisher: Callable = None):
        self.config = config
        self._push_cache = None
        self._push_alert_timestamps = []
        self._push_last_sent_ts = 0.0
        self._push_last_state = False
        self.publisher = publisher
        self.state_get = None
        self.state_set = None
        # set up workflow helpers
        mqtt_cfg = self.config.get('MQTTPUBLISH', {})
        # slot_map is a mapping of canonical sensor names -> payload keys in incoming MQTT data
        slot_map = mqtt_cfg.get('PUSH_SLOT_MAP') or {
            'rain': 'rain',
            'humidity': 'humidity',
            'pressure': 'pressure',
            'dew_point': 'dew_point',
            'lightning': 'lightning',
        }
        self.data_packager = DataPackager(slot_map)
        self.analysis = AnalysisWorkflow()
        time_gate_cfg = mqtt_cfg.get('PUSH_TIME_GATE', {})
        tg = TimeGate(enabled=time_gate_cfg.get('enabled', False),
                      start_hour=time_gate_cfg.get('start_hour', 0),
                      end_hour=time_gate_cfg.get('end_hour', 24))
        self.push_messaging = PushMessaging(self.publisher, time_gate=tg)

    def _call_publisher(self, topic, payload):
        if not self.publisher:
            return
        try:
            if callable(self.publisher):
                self.publisher(topic, payload)
            elif hasattr(self.publisher, 'publish'):
                self.publisher.publish(topic, payload)
        except Exception:
            logger.exception('Error publishing push alert')

    def evaluate_and_send(self, mqtt_data: dict):
        # configuration defaults
        mqtt_cfg = self.config.get('MQTTPUBLISH', {})
        hours = float(mqtt_cfg.get('PUSH_HISTORY_HOURS', 3.0))
        seconds_window = int(hours * 3600)
        max_entries = int(mqtt_cfg.get('PUSH_MAX_ENTRIES', 1000))

        # ensure cache exists
        if self._push_cache is None:
            self._push_cache = PushHistoryCache(max_entries=max_entries)
        else:
            self._push_cache.resize(max_entries)

        now = time.time()
        # append incoming sample and prune by time window
        self._push_cache.append(now, mqtt_data)
        self._push_cache.prune(seconds_window)
        history = self._push_cache.get_recent(seconds_window)

        # Decide whether to use analysis workflow or legacy expression model
        use_analysis = bool(mqtt_cfg.get('PUSH_USE_ANALYSIS', False))
        triggered = False
        cooldown = int(mqtt_cfg.get('PUSH_COOLDOWN_S', 0))
        max_per_hour = int(mqtt_cfg.get('PUSH_MAX_PER_HOUR', 0))

        if use_analysis:
            # Build sensor histories from the log-history and run analysis
            try:
                sensor_histories = self.data_packager.build(history, window_s=seconds_window)
                # pass spike detection params through config if provided
                spike_kwargs = {
                    'long_window_s': int(mqtt_cfg.get('PUSH_LONG_WINDOW_S', 3 * 3600)),
                    'short_window_s': int(mqtt_cfg.get('PUSH_SHORT_WINDOW_S', 30 * 60)),
                    'spike_multiplier': float(mqtt_cfg.get('PUSH_SPIKE_MULTIPLIER', 2.0)),
                    'absolute_delta': mqtt_cfg.get('PUSH_SPIKE_ABSOLUTE_DELTA', None),
                }
                score_obj = self.analysis.analyze(
                    sensor_histories,
                    sensor_weights=mqtt_cfg.get('PUSH_SENSOR_WEIGHTS', None),
                    include=mqtt_cfg.get('PUSH_INCLUDE_SENSORS', None),
                    feature_key=mqtt_cfg.get('PUSH_FEATURE_KEY', rdm.DEFAULT_FEATURE_KEY) if 'rdm' in globals() else mqtt_cfg.get('PUSH_FEATURE_KEY', 'overall_mean'),
                    normalize=bool(mqtt_cfg.get('PUSH_NORMALIZE', True)),
                    **spike_kwargs,
                )
                threshold = float(mqtt_cfg.get('PUSH_SCORE_THRESHOLD', 0.0))
                triggered = float(score_obj.get('score', 0.0)) >= threshold
            except Exception:
                logger.exception('AnalysisWorkflow evaluation failed')
                triggered = False
        else:
            model = str(mqtt_cfg.get('PUSH_MODEL', '')).strip()
            if not model:
                return
            # bind small helpers to current history for expression usage
            trend_slope = lambda slot, window_s=None: util_trend_slope(history, slot, window_s)
            increasing = lambda slot, window_s=None, min_slope=0.0: util_increasing(history, slot, window_s, min_slope)
            decreasing = lambda slot, window_s=None, min_slope=0.0: util_decreasing(history, slot, window_s, min_slope)

            try:
                ctx = {
                    'history': history,
                    'current': mqtt_data,
                    'time': time,
                    'trend_slope': trend_slope,
                    'increasing': increasing,
                    'decreasing': decreasing,
                    'PUSH_HISTORY_HOURS': hours,
                    'PUSH_HISTORY_SECONDS': seconds_window,
                    'PUSH_COOLDOWN_S': cooldown,
                    'PUSH_MAX_PER_HOUR': max_per_hour,
                }
                allowed_calls = {'trend_slope', 'increasing', 'decreasing'}
                triggered = bool(_safe_eval(model, ctx, allowed_calls))
            except UnsafeExpression as e:
                logger.error('Push model unsafe: %s', e)
                triggered = False
            except Exception as e:
                logger.error('Push model evaluation error: %s', e)
                triggered = False

        # load persisted state if available
        try:
            if self.state_get:
                last_sent_s = self.state_get('PUSH_LAST_SENT_TS')
                if last_sent_s:
                    self._push_last_sent_ts = float(last_sent_s)
                sent_list_s = self.state_get('PUSH_ALERT_TIMESTAMPS')
                if sent_list_s:
                    self._push_alert_timestamps = json.loads(sent_list_s)
        except Exception:
            pass

        last_state = self._push_last_state

        if triggered and not last_state:
            now_ts = time.time()
            last_sent = float(getattr(self, '_push_last_sent_ts', 0.0))
            if cooldown and (now_ts - last_sent) < cooldown:
                logger.info('Push alert suppressed by cooldown (%ds remaining)', int(cooldown - (now_ts - last_sent)))
            else:
                sent_list = list(getattr(self, '_push_alert_timestamps', []))
                hour_cutoff = now_ts - 3600
                sent_list = [t for t in sent_list if t >= hour_cutoff]
                if max_per_hour and len(sent_list) >= max_per_hour:
                    logger.info('Push alert suppressed by hourly limit (%d/hour)', max_per_hour)
                else:
                    alert_topic = mqtt_cfg.get('PUSH_TOPIC', 'alert')
                    alert_data = {'alert': 'push', 'timestamp': datetime.utcnow().isoformat()}
                    alert_data.update(mqtt_data)
                    # use PushMessaging to respect configured time gate
                    try:
                        self.push_messaging.send(alert_topic, alert_data)
                    except Exception:
                        # fallback to direct call
                        self._call_publisher(alert_topic, alert_data)
                    sent_list.append(now_ts)
                    self._push_alert_timestamps = sent_list
                    self._push_last_sent_ts = now_ts
                    try:
                        if self.state_set:
                            self.state_set('PUSH_ALERT_TIMESTAMPS', json.dumps(self._push_alert_timestamps))
                            self.state_set('PUSH_LAST_SENT_TS', str(self._push_last_sent_ts))
                    except Exception:
                        logger.exception('Failed to persist push state')

        self._push_last_state = triggered
