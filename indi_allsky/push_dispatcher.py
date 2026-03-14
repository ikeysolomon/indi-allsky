"""Push evaluator, publisher and dispatch worker.

Integration notes
-----------------
This module consumes processed items from the dispatch queue provided by
`weather_station` (imported here as `metric_queue`). It exposes `PushEvaluator`
and `MQTTPublisher` and a `run()` worker suitable for running in a separate
process. `PushEvaluator` persists suppression counters (cooldown/hourly/
timegate/publish_error) via the configured `state_get`/`state_set` handlers
when available.

Configuration
-------------
Tune behaviour through the `MQTTPUBLISH` config namespace (examples:
`PUSH_COOLDOWN_S`, `PUSH_MAX_PER_HOUR`, `PUSH_USE_ANALYSIS`). `MQTTPublisher`
enqueues a DB task for actual MQTT delivery; retained publish flags are not
set by default (add `PUSH_MQTT_RETAIN` handling in the uploader to change
that behaviour).
"""
import time
import logging
import ast
import json
import traceback
from datetime import datetime
from typing import Callable, Optional

# Only import the specific queue API used by this module (avoids loading the full
# module namespace if not needed elsewhere).
from .weather_station import get_dispatch

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
        # Lazy-import DB/task queue dependencies to avoid heavy imports at
        # module import time. Publishing is an infrequent, I/O-bound action so
        # delaying imports until needed is acceptable.
        if self.config and not self.config.get('MQTTPUBLISH', {}).get('ENABLE'):
            return
        if not self.upload_q:
            logger.warning('No upload queue available for MQTTPublisher')
            return
        try:
            from .flask import db
            from .flask.models import IndiAllSkyDbTaskQueueTable
            from .flask.models import TaskQueueQueue, TaskQueueState
            from . import constants

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


class PushMessaging:
    def __init__(self, publisher, time_gate: Optional[object] = None):
        self.publisher = publisher
        self.time_gate = time_gate

    def send(self, topic: str, payload: dict) -> None:
        """Send payload via configured publisher.

        Returns True if the payload was handed off for publish, False if it
        was suppressed by the time gate or failed to publish.
        """
        try:
            if self.time_gate and not getattr(self.time_gate, 'is_open', lambda: True)():
                logger.info('Push suppressed by time gate')
                return False
            if callable(self.publisher):
                self.publisher(topic, payload)
            elif hasattr(self.publisher, 'publish'):
                self.publisher.publish(topic, payload)
            else:
                logger.debug('No publisher configured for PushMessaging')
            return True
        except Exception:
            logger.exception('PushMessaging: error publishing')
            return False


class PushEvaluator:
    def __init__(self, config: dict, publisher: Callable = None, history_packager: Callable = None):
        self.config = config
        self._push_cache = None
        self._push_alert_timestamps = []
        self._push_last_sent_ts = 0.0
        self._push_last_state = False
        self.publisher = publisher
        self.state_get = None
        self.state_set = None
        # suppression counters for telemetry/auditing
        self.suppressed_counters = {'cooldown': 0, 'hourly': 0, 'timegate': 0, 'publish_error': 0}
        mqtt_cfg = self.config.get('MQTTPUBLISH', {})
        time_gate_cfg = mqtt_cfg.get('PUSH_TIME_GATE', {})
        tg = TimeGate(enabled=time_gate_cfg.get('enabled', False),
                      start_hour=time_gate_cfg.get('start_hour', 0),
                      end_hour=time_gate_cfg.get('end_hour', 24))
        self.push_messaging = PushMessaging(self.publisher, time_gate=tg)
        self.history_packager = history_packager
        # Lazy-loaded default history packager; imported only when needed.
        self._default_packager = None

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

    def _persist_suppressed_counters(self):
        try:
            if self.state_set:
                self.state_set('PUSH_SUPPRESSED_COUNTERS', json.dumps(self.suppressed_counters))
        except Exception:
            logger.exception('Failed to persist suppressed counters')

    def set_state_handlers(self, getter, setter):
        self.state_get = getter
        self.state_set = setter
        # attempt to restore persisted suppressed counters
        try:
            if self.state_get:
                val = self.state_get('PUSH_SUPPRESSED_COUNTERS')
                if val:
                    self.suppressed_counters = json.loads(val)
        except Exception:
            logger.exception('Failed to restore suppressed counters')

    def evaluate_and_send(self, mqtt_data: dict):
        mqtt_cfg = self.config.get('MQTTPUBLISH', {})
        hours = float(mqtt_cfg.get('PUSH_HISTORY_HOURS', 3.0))
        seconds_window = int(hours * 3600)
        max_entries = int(mqtt_cfg.get('PUSH_MAX_ENTRIES', 1000))

        use_analysis = bool(mqtt_cfg.get('PUSH_USE_ANALYSIS', False))

        # Align all sensor windows/series to the same "now" reference.
        now_ts = time.time()

        try:
            if self.history_packager:
                packaged = self.history_packager(mqtt_data, mqtt_cfg, now_ts=now_ts)
            else:
                # If the incoming `mqtt_data` already contains packaged results
                # (e.g. produced by the Meteorologist worker), treat it as the
                # packaged dict and avoid recomputing heavy analysis here.
                if isinstance(mqtt_data, dict) and ('sensor_histories' in mqtt_data or 'history' in mqtt_data):
                    packaged = mqtt_data
                else:
                    if self._default_packager is None:
                        from .Meteorologist import append_sample_and_build_histories as _default_packager
                        self._default_packager = _default_packager
                    packaged = self._default_packager(mqtt_data, mqtt_cfg, now_ts=now_ts)
            history = packaged.get('history', [])
            sensor_histories = packaged.get('sensor_histories', {})
        except Exception:
            logger.exception('Failed building histories via packager')
            packaged = {}
            history = []
            sensor_histories = {}
        triggered = False
        # sensible defaults: 3 minutes cooldown, 6 alerts per hour
        cooldown = int(mqtt_cfg.get('PUSH_COOLDOWN_S', 180))
        max_per_hour = int(mqtt_cfg.get('PUSH_MAX_PER_HOUR', 6))

        # If upstream has already computed a push decision/score, use it.
        if isinstance(packaged, dict) and 'push_triggered' in packaged:
            try:
                triggered = bool(packaged.get('push_triggered'))
            except Exception:
                triggered = False
        elif isinstance(packaged, dict) and 'push_score' in packaged:
            try:
                triggered = float(packaged.get('push_score', 0.0)) >= float(mqtt_cfg.get('PUSH_SCORE_THRESHOLD', 0.0))
            except Exception:
                triggered = False
        else:
            # No precomputed decision available from upstream analysis. The
            # dispatcher must remain lightweight and must not perform heavy
            # analysis itself. Skip push evaluation and log for auditing.
            logger.debug('No precomputed push decision in packaged data; skipping evaluation in dispatcher')
            triggered = False

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
            last_sent = float(getattr(self, '_push_last_sent_ts', 0.0))
            if cooldown and (now_ts - last_sent) < cooldown:
                remaining = int(cooldown - (now_ts - last_sent))
                logger.info('Push alert suppressed by cooldown (%ds remaining)', remaining)
                try:
                    self.suppressed_counters['cooldown'] += 1
                    self._persist_suppressed_counters()
                except Exception:
                    pass
            else:
                sent_list = list(getattr(self, '_push_alert_timestamps', []))
                hour_cutoff = now_ts - 3600
                sent_list = [t for t in sent_list if t >= hour_cutoff]
                if max_per_hour and len(sent_list) >= max_per_hour:
                    logger.info('Push alert suppressed by hourly limit (%d/hour)', max_per_hour)
                    try:
                        self.suppressed_counters['hourly'] += 1
                        self._persist_suppressed_counters()
                    except Exception:
                        pass
                else:
                    alert_topic = mqtt_cfg.get('PUSH_TOPIC', 'alert')
                    alert_data = {'alert': 'push', 'timestamp': datetime.utcnow().isoformat()}
                    alert_data.update(mqtt_data)
                    try:
                        alert_data['high_cloud'] = bool(packaged.get('high_cloud'))
                    except Exception:
                        alert_data['high_cloud'] = False
                    try:
                        sent = self.push_messaging.send(alert_topic, alert_data)
                        if not sent:
                            # suppressed by timegate or failed to publish
                            try:
                                self.suppressed_counters['timegate'] += 1
                                self._persist_suppressed_counters()
                            except Exception:
                                pass
                    except Exception:
                        try:
                            self.suppressed_counters['publish_error'] += 1
                            self._persist_suppressed_counters()
                        except Exception:
                            pass
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


def send_payload_log(item: dict):
    logger.info('push_dispatcher: dispatching payload: %s', item)


def run(mqtt_cfg: dict, shutdown_event: Optional[object] = None):
    if shutdown_event is None:
        # in-process simple sentinel
        class _S:
            def is_set(self):
                return False
        shutdown_event = _S()

    # create a PushEvaluator with a no-op publisher by default
    publisher = None
    try:
        mqtt_cfg = mqtt_cfg or {}
        push_eval = PushEvaluator(mqtt_cfg, publisher=publisher)
    except Exception:
        logger.exception('Failed creating PushEvaluator')
        push_eval = None

    logger.info('push_dispatcher: starting loop')
    while not getattr(shutdown_event, 'is_set', lambda: False)():
        try:
            item = get_dispatch(block=True, timeout=1)
        except Exception:
            continue

        try:
            # prefer the raw metric data for evaluation
            mqtt_data = None
            if isinstance(item, dict):
                mqtt_data = item.get('raw', {}).get('data') or item.get('result')
            if push_eval and mqtt_data is not None:
                try:
                    push_eval.evaluate_and_send(mqtt_data)
                except Exception:
                    logger.exception('push_dispatcher: evaluation failed')
            else:
                send_payload_log(item)
        except Exception:
            logger.error('push_dispatcher: error while dispatching: %s', traceback.format_exc())

    logger.info('push_dispatcher: exiting')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run({}, None)
