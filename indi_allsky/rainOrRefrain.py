"""Encapsulate push-model evaluation and rate-limiting.

This class manages a small in-memory history for incoming MQTT data,
evaluates a user-configured predicate (a safe expression) and calls a
publisher when the predicate transitions true->false according to
cooldown and hourly limits.

Design notes:
- This module is a light wrapper that receives analysis signals from
    `rain_detection_models`. The analysis module manages history storage
    (using `MoistureMonitorCache`) and performs heavy computation.
    `rainOrRefrain` only handles gating and publishing.
"""

import time
import logging
import ast
import json
from datetime import datetime
from typing import Callable, Optional

from . import rain_detection_models as rdm


class PushMessaging:
    """Publish alerts subject to a `TimeGate`.

    `publisher` may be a callable (topic,payload) or an object with
    `publish(topic,payload)`.
    """

    def __init__(self, publisher, time_gate: Optional[object] = None):
        self.publisher = publisher
        self.time_gate = time_gate

    def send(self, topic: str, payload: dict) -> None:
        try:
            if self.time_gate and not getattr(self.time_gate, 'is_open', lambda: True)():
                logger.info('Push suppressed by time gate')
                return
            if callable(self.publisher):
                self.publisher(topic, payload)
            elif hasattr(self.publisher, 'publish'):
                self.publisher.publish(topic, payload)
        except Exception:
            logger.exception('PushMessaging: error publishing')
from .flask import db
from .flask.models import IndiAllSkyDbTaskQueueTable
from .flask.models import TaskQueueQueue, TaskQueueState
from . import constants

logger = logging.getLogger('indi_allsky')


class NotificationPublisher:
    def publish(self, topic: str, data):
        raise NotImplementedError()


class MQTTPublisher(NotificationPublisher):
    def __init__(self, misc_upload=None, upload_q=None, config=None):
        self.misc_upload = misc_upload
        self.upload_q = upload_q or (getattr(misc_upload, 'upload_q', None) if misc_upload else None)
        self.config = config

    def publish(self, topic: str, data):
        if self.config and not self.config.get('MQTTPUBLISH', {}).get('ENABLE'):
            return
        if not self.upload_q:
            logger.warning('No upload queue available for MQTTPublisher')
            return
        try:
            jobdata = {
                'action': constants.TRANSFER_MQTT,
                'local_file': '',
                'image_topic': topic,
                'metadata': data,
                'publish_image': False,
            }

            mqtt_task = IndiAllSkyDbTaskQueueTable(
                queue=TaskQueueQueue.UPLOAD,
                state=TaskQueueState.QUEUED,
                data=jobdata,
            )
            db.session.add(mqtt_task)
            db.session.commit()

            self.upload_q.put({'task_id': mqtt_task.id})
        except Exception:
            logger.exception('Failed to enqueue mqtt publish task')


def get_series(history, slot: str, window_s: Optional[float] = None, now_ts: Optional[float] = None):
    """Return a list of (ts, value) tuples for numeric values of `slot` in history.

    Accepts the same history format used by `MoistureMonitorCache.get_recent()`.
    """
    if now_ts is None:
        now_ts = __import__('time').time()
    cutoff_ts = now_ts - window_s if window_s else 0
    series = []
    for ts, d in history:
        if ts < cutoff_ts:
            continue
        if isinstance(d, dict) and slot in d and isinstance(d[slot], (int, float)):
            series.append((ts, float(d[slot])))
    return series


class TimeGate:
    """Simple UI-configurable time gate used by push logic.

    Kept here to centralize time-related gating used by `PushEvaluator`.
    """
    def __init__(self, enabled: bool = False, start_hour: int = 0, end_hour: int = 24):
        self.enabled = bool(enabled)
        self.start_hour = int(start_hour)
        self.end_hour = int(end_hour)

    def is_open(self, now_ts: Optional[float] = None) -> bool:
        if not self.enabled:
            return True
        t = time.localtime(now_ts) if now_ts is not None else time.localtime()
        h = t.tm_hour
        if self.start_hour <= self.end_hour:
            return self.start_hour <= h < self.end_hour
        return h >= self.start_hour or h < self.end_hour


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
    def __init__(self, config: dict, publisher: Callable = None, history_packager: Callable = None):
        self.config = config
        # rainOrRefrain no longer owns a cache; analysis module manages it.
        self._push_cache = None
        self._push_alert_timestamps = []
        self._push_last_sent_ts = 0.0
        self._push_last_state = False
        self.publisher = publisher
        self.state_get = None
        self.state_set = None
        # set up workflow helpers
        mqtt_cfg = self.config.get('MQTTPUBLISH', {})
        # history/packaging is handled by Meteorologist
        time_gate_cfg = mqtt_cfg.get('PUSH_TIME_GATE', {})
        tg = TimeGate(enabled=time_gate_cfg.get('enabled', False),
                      start_hour=time_gate_cfg.get('start_hour', 0),
                      end_hour=time_gate_cfg.get('end_hour', 24))
        self.push_messaging = PushMessaging(self.publisher, time_gate=tg)
        # Optional injectable packager callable: (mqtt_data, mqtt_cfg, now_ts) -> packaged dict
        # If not provided, a lazy import fallback is used inside `evaluate_and_send`.
        self.history_packager = history_packager

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

        # Decide whether to use analysis workflow or legacy expression model
        use_analysis = bool(mqtt_cfg.get('PUSH_USE_ANALYSIS', False))

        # Build histories via injected packager or lazy-imported Meteorologist
        try:
            if self.history_packager:
                packaged = self.history_packager(mqtt_data, mqtt_cfg, now_ts=time.time())
            else:
                # keep a lazy fallback to avoid a hard dependency at module import time
                from .Meteorologist import append_sample_and_build_histories as _default_packager
                packaged = _default_packager(mqtt_data, mqtt_cfg, now_ts=time.time())
            history = packaged.get('history', [])
            sensor_histories = packaged.get('sensor_histories', {})
        except Exception:
            logger.exception('Failed building histories via packager')
            history = []
            sensor_histories = {}
        triggered = False
        cooldown = int(mqtt_cfg.get('PUSH_COOLDOWN_S', 0))
        max_per_hour = int(mqtt_cfg.get('PUSH_MAX_PER_HOUR', 0))

        if use_analysis:
            try:
                score_obj = rdm.compute_weighted_rain_score_with_recent_priority(
                    sensor_histories,
                    long_window_s=int(float(mqtt_cfg.get('PUSH_LONG_WINDOW_S', 3 * 3600))),
                    short_window_s=int(float(mqtt_cfg.get('PUSH_SHORT_WINDOW_S', 30 * 60))),
                    spike_multiplier=float(mqtt_cfg.get('PUSH_SPIKE_MULTIPLIER', 2.0)),
                    absolute_delta=float(mqtt_cfg.get('PUSH_SPIKE_ABSOLUTE_DELTA', 0.5)),
                    weights=mqtt_cfg.get('PUSH_SENSOR_WEIGHTS') or None,
                    # pass star/cloud signals produced by Meteorologist
                    star_count=packaged.get('star_count'),
                    high_cloud=packaged.get('high_cloud_value') or packaged.get('high_cloud'),
                    high_cloud_multiplier=float(mqtt_cfg.get('PUSH_HIGH_CLOUD_MULTIPLIER', 1.2)),
                    # slope-derived signal config
                    slope_window_s=int(float(mqtt_cfg.get('PUSH_SLOPE_WINDOW_S', 30 * 60))),
                    hum_delta_norm=float(mqtt_cfg.get('PUSH_HUM_DELTA_NORM', 0.5)),
                    pres_delta_norm=float(mqtt_cfg.get('PUSH_PRES_DELTA_NORM', 1.0)),
                    hum_slope_weight=float(mqtt_cfg.get('PUSH_HUM_SLOPE_WEIGHT', 0.25)),
                    pres_slope_weight=float(mqtt_cfg.get('PUSH_PRES_SLOPE_WEIGHT', 0.25)),
                    slope_combined_multiplier=float(mqtt_cfg.get('PUSH_SLOPE_COMBINED_MULTIPLIER', 1.3)),
                    slope_signal_threshold=float(mqtt_cfg.get('PUSH_SLOPE_SIGNAL_THRESHOLD', 0.5)),
                )
                triggered = float(score_obj.get('score', 0.0)) >= float(mqtt_cfg.get('PUSH_SCORE_THRESHOLD', 0.0))
            except Exception:
                logger.exception('rainOrRefrain: analysis failed')
                triggered = False
        else:
            model = str(mqtt_cfg.get('PUSH_MODEL', '')).strip()
            if not model:
                return
            # bind small helpers to current history for expression usage
            trend_slope = lambda slot, window_s=None: rdm.linear_slope(get_series(history, slot, window_s))
            increasing = lambda slot, window_s=None, min_slope=0.0: (trend_slope(slot, window_s) or 0.0) >= float(min_slope)
            decreasing = lambda slot, window_s=None, min_slope=0.0: (trend_slope(slot, window_s) or 0.0) <= -float(min_slope)

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
                    # include high-cloud signal from analysis packaging
                    try:
                        alert_data['high_cloud'] = bool(packaged.get('high_cloud'))
                    except Exception:
                        alert_data['high_cloud'] = False
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
