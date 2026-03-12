"""Encapsulate push-model evaluation and rate-limiting.

This class centralises history management, model evaluation, cooldown and
hourly rate limiting. It can call a publisher to send alerts but does not
depend on the upload/task queue directly.
"""
import time
from datetime import datetime
import logging
import ast
from typing import Callable

import numpy

from .push_history import PushHistoryCache

logger = logging.getLogger('indi_allsky')


class UnsafeExpression(Exception):
    pass


def _safe_eval(expr: str, ctx: dict, allowed_calls: set):
    """Evaluate a user expression with basic safety checks.

    - Parses the expression to AST and rejects nodes that look unsafe (e.g. attribute
      assignment, import, etc.).
    - Allows function calls only when the function name is in `allowed_calls`.
    """
    node = ast.parse(expr, mode='eval')

    for n in ast.walk(node):
        # Disallow anything that can mutate or import
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.Delete, ast.Lambda, ast.Global, ast.With, ast.Try)):
            raise UnsafeExpression('Forbidden AST node')
        if isinstance(n, ast.Call):
            # Only allow simple name calls (no attribute calls) and only whitelisted names
            func = n.func
            if isinstance(func, ast.Name):
                if func.id not in allowed_calls:
                    raise UnsafeExpression('Call to disallowed function')
            else:
                raise UnsafeExpression('Only simple function names are allowed')

    # use eval with restricted builtins
    return eval(compile(node, '<push_model>', 'eval'), {'__builtins__': None}, ctx)


class PushEvaluator:
    def __init__(self, config: dict, publisher: Callable = None):
        self.config = config
        self._push_cache = None
        self._push_alert_timestamps = []
        self._push_last_sent_ts = 0
        self._push_last_state = False
        # publisher can be a callable (topic, data) or an object with publish()
        self.publisher = publisher

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
        if not self.config.get('MQTTPUBLISH', {}).get('PUSH_ENABLE', False):
            return

        now = int(time.time())

        hours = int(self.config.get('MQTTPUBLISH', {}).get('PUSH_HISTORY_HOURS', 0))
        seconds_window = hours * 3600

        # derive max_entries heuristically or fallback to config
        exposure_max = float(self.config.get('EXPOSURE_PERIOD', 60))
        raw_entries = seconds_window / max(1.0, exposure_max)
        max_entries = int(max(10, (raw_entries * 1.2)))
        cfg_override = self.config.get('MQTTPUBLISH', {}).get('PUSH_HISTORY_MAX_ENTRIES')
        if cfg_override:
            try:
                max_entries = int(cfg_override)
            except Exception:
                pass

        if self._push_cache is None:
            self._push_cache = PushHistoryCache(max_entries=max_entries)
        else:
            self._push_cache.resize(max_entries)

        # append and prune
        self._push_cache.append(now, mqtt_data)
        self._push_cache.prune(seconds_window)
        history = self._push_cache.get_recent(seconds_window)

        model = str(self.config.get('MQTTPUBLISH', {}).get('PUSH_MODEL', '')).strip()
        if not model:
            return

        # helper functions for model context
        def _get_series(slot, window_s=None):
            now_ts = time.time()
            series = []
            cutoff_ts = now_ts - window_s if window_s else 0
            for ts, d in history:
                if ts < cutoff_ts:
                    continue
                if slot in d and isinstance(d[slot], (int, float)):
                    series.append((ts, float(d[slot])))
            return series

        def trend_slope(slot, window_s=None):
            s = _get_series(slot, window_s)
            if len(s) < 2:
                return 0.0
            xs = numpy.array([int(t) for t, v in s], dtype=float)
            ys = numpy.array([v for t, v in s], dtype=float)
            xs0 = xs - xs[0]
            try:
                p = numpy.polyfit(xs0, ys, 1)
                slope_per_second = float(p[0])
                slope = slope_per_second * 60.0
            except Exception:
                slope = 0.0
            return slope

        def increasing(slot, window_s=None, min_slope=0.0):
            return trend_slope(slot, window_s) > float(min_slope)

        def decreasing(slot, window_s=None, min_slope=0.0):
            return trend_slope(slot, window_s) < -float(min_slope)

        cooldown = int(self.config.get('MQTTPUBLISH', {}).get('PUSH_COOLDOWN_S', 0))
        max_per_hour = int(self.config.get('MQTTPUBLISH', {}).get('PUSH_MAX_PER_HOUR', 0))

        triggered = False
        try:
            ctx = {
                'history': history,
                'current': mqtt_data,
                'time': time,
                'trend_slope': trend_slope,
                'increasing': increasing,
                'decreasing': decreasing,
                'PUSH_HISTORY_HOURS': hours,
                'PUSH_HISTORY_SECONDS': int(hours * 3600),
                'PUSH_COOLDOWN_S': cooldown,
                'PUSH_MAX_PER_HOUR': max_per_hour,
            }
            # allow only these helper functions to be called from expressions
            allowed_calls = {'trend_slope', 'increasing', 'decreasing'}
            triggered = bool(_safe_eval(model, ctx, allowed_calls))
        except UnsafeExpression as e:
            logger.error('Push model unsafe: %s', e)
            triggered = False
        except Exception as e:
            logger.error('Push model evaluation error: %s', e)
            triggered = False

        last_state = self._push_last_state

        if triggered and not last_state:
            now_ts = time.time()
            last_sent = getattr(self, '_push_last_sent_ts', 0)
            if cooldown and (now_ts - last_sent) < cooldown:
                logger.info('Push alert suppressed by cooldown (%ds remaining)', int(cooldown - (now_ts - last_sent)))
            else:
                sent_list = getattr(self, '_push_alert_timestamps', [])
                hour_cutoff = now_ts - 3600
                sent_list = [t for t in sent_list if t >= hour_cutoff]
                if max_per_hour and len(sent_list) >= max_per_hour:
                    logger.info('Push alert suppressed by hourly limit (%d/hour)', max_per_hour)
                else:
                    alert_topic = self.config.get('MQTTPUBLISH', {}).get('PUSH_TOPIC', 'alert')
                    alert_data = {'alert': 'push', 'timestamp': datetime.utcnow().isoformat()}
                    alert_data.update(mqtt_data)
                    self._call_publisher(alert_topic, alert_data)
                    sent_list.append(now_ts)
                    self._push_alert_timestamps = sent_list
                    self._push_last_sent_ts = now_ts

        self._push_last_state = triggered
