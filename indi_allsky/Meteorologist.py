"""Meteorologist: history scanning, caching and packaging for analysis.

Integration notes
-----------------
When `MQTTPUBLISH.PUSH_ASYNC` is enabled, `append_sample_and_build_histories`
will enqueue incoming metrics into the process-safe queue provided by
`weather_station` (imported internally as `metric_queue`). This module also
exposes `run_worker()` for running the consumer loop in a separate process.

Responsibilities:
- Maintain a bounded time-windowed cache for incoming MQTT samples.
- Provide `DataPackager` functionality to build `sensor_histories`.
- Provide minimal date calcs used by capture/processing.
- Compute a concise, graded high-cloud signal (and boolean flag) from
    available star-count data so analysis can treat cloudiness as a feature.

This module centralizes I/O and state (cache/packaging) so the analysis
module (`rain_detection_models`) can remain pure and focused on signal
generation. Heavy analysis models are not imported here to keep runtime
coupling low; callers may lazily import or inject analysis functions.
"""

# --- Hoisted configuration defaults (module-level constants) ---
# These may be overridden via the `MQTTPUBLISH` config passed into
# `append_sample_and_build_histories`.
DEFAULT_HIGH_CLOUD_STAR_THRESHOLD = 30
DEFAULT_HIGH_CLOUD_AFTER_HOUR = 17
from typing import Dict, Iterable, List, Optional
from datetime import datetime, timedelta

# Prefer the project's canonical date calcs from `utils.py` (ephem-based).
# Lazy-load the heavy `utils` implementation to avoid importing `ephem`
# at module import time.
_IndiAllSkyDateCalcs_cls = None
def _get_IndiAllSkyDateCalcs():
    global _IndiAllSkyDateCalcs_cls
    if _IndiAllSkyDateCalcs_cls is None:
        from .utils import IndiAllSkyDateCalcs as _Cls
        _IndiAllSkyDateCalcs_cls = _Cls
    return _IndiAllSkyDateCalcs_cls

def IndiAllSkyDateCalcs(*args, **kwargs):
    """Factory that lazily instantiates the canonical IndiAllSkyDateCalcs.

    Call like a class: `IndiAllSkyDateCalcs(config, position_av)` -> instance.
    """
    cls = _get_IndiAllSkyDateCalcs()
    return cls(*args, **kwargs)

class MoistureMonitorCache:
    """Bounded, deque-backed time-windowed cache for push history."""
    def __init__(self, max_entries=None):
        from collections import deque
        self.max_entries = int(max_entries) if max_entries else None
        self._entries = deque(maxlen=self.max_entries)

    def append(self, ts, data):
        self._entries.append((int(ts), data.copy()))

    def get_recent(self, seconds_window):
        import time
        if seconds_window <= 0:
            return list(self._entries)
        cutoff = int(time.time()) - int(seconds_window)
        return [(ts, d) for ts, d in self._entries if ts >= cutoff]

    def prune(self, seconds_window, now_ts: Optional[float] = None):
        import time
        if seconds_window <= 0:
            return
        ref = int(now_ts) if now_ts is not None else int(time.time())
        cutoff = int(ref) - int(seconds_window)
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def resize(self, max_entries):
        max_entries = int(max_entries) if max_entries else None
        if max_entries == self.max_entries:
            return
        entries = list(self._entries)
        self.max_entries = max_entries
        from collections import deque
        self._entries = deque(entries, maxlen=self.max_entries)


# `IndiAllSkyDateCalcs` is provided by `indi_allsky.utils` (imported above).
# A thin wrapper used to be here; prefer the canonical implementation there.


class DataPackager:
    """Convert a log-history of (ts,payload) into per-sensor histories.

    build(history, window_s) -> (sensor_histories_raw, sensor_histories_norm)
    """

    def __init__(self, slot_map: Dict[str, str], sensor_ranges: Optional[Dict[str, tuple]] = None):
        self.slot_map = dict(slot_map)
        # sensor_ranges: mapping sensor -> (min, max) for normalization
        # sensible defaults are applied when not provided
        self.sensor_ranges = sensor_ranges or {
            'pressure': (900.0, 1100.0),
            'humidity': (0.0, 100.0),
            'lightning': (0.0, 1.0),
            'rain': (0.0, 1.0),
            'temperature': (-30.0, 50.0),
            'dew_point': (-30.0, 30.0),
        }

    def build(self, history: Iterable, window_s: Optional[float] = None, now_ts: Optional[float] = None):
        """Return (sensor_histories_raw, sensor_histories_norm).

        - sensor_histories_raw: mapping sensor -> list[(ts, raw_value)]
        - sensor_histories_norm: mapping sensor -> list[(ts, norm_value)] where
            norm_value is clamped to [0.0, 1.0] using configured ranges. If a
            sensor has no configured range, the normalized series is returned
            as an empty list and a debug message is emitted.
        """
        import logging

        LOG = logging.getLogger('meteorologist.packager')

        sensor_histories: Dict[str, List] = {}
        sensor_histories_norm: Dict[str, List] = {}

        # materialize history once to avoid repeated copies
        history_list = list(history)

        # Determine a reference 'now' timestamp. If the caller provided an
        # explicit `now_ts`, use it so all sensors and related pruning/window
        # logic share a single temporal reference. Otherwise, fall back to the
        # most-recent history entry timestamp (if available) or wall-clock.
        if now_ts is None:
            try:
                if history_list:
                    now_ts = int(history_list[-1][0])
                else:
                    now_ts = None
            except Exception:
                now_ts = None
        else:
            try:
                now_ts = int(now_ts)
            except Exception:
                now_ts = None

        for sensor, slot in self.slot_map.items():
            series: List = []
            for ts, payload in history_list:
                if window_s is not None and now_ts is not None and ts < now_ts - window_s:
                    continue
                if isinstance(payload, dict) and slot in payload and isinstance(payload[slot], (int, float)):
                    series.append((ts, float(payload[slot])))
            sensor_histories[sensor] = series

            # build normalized series using configured ranges
            rng = self.sensor_ranges.get(sensor)
            norm_series: List = []

            # Special-case `rain` as a boolean presence flag: map any
            # measured precipitation (>0) to 1.0, otherwise 0.0. This keeps
            # rain from dominating continuous weighted scores.
            if sensor == 'rain' and series:
                try:
                    for ts, raw in series:
                        try:
                            v = float(raw)
                        except Exception:
                            v = 0.0
                        norm_series.append((ts, 1.0 if v > 0.0 else 0.0))
                except Exception:
                    norm_series = []
            elif rng is not None and series:
                try:
                    mn, mx = float(rng[0]), float(rng[1])
                    span = mx - mn if mx != mn else None
                    for ts, raw in series:
                        if span:
                            norm = (float(raw) - mn) / span
                        else:
                            norm = 0.0
                        # clamp
                        if norm != norm:  # NaN guard
                            norm = 0.0
                        norm = max(0.0, min(1.0, float(norm)))
                        norm_series.append((ts, norm))
                except Exception:
                    norm_series = []
            else:
                # no range configured -> return empty normalized series and
                # emit a debug message so the caller can decide how to handle it
                if rng is None and series:
                    LOG.debug('No sensor range configured for "%s"; normalized series empty', sensor)
                norm_series = []

            sensor_histories_norm[sensor] = norm_series

        return sensor_histories, sensor_histories_norm


# module-level singleton cache
_cache: Optional[MoistureMonitorCache] = None


def _ensure_cache(max_entries: int):
    global _cache
    if _cache is None or getattr(_cache, 'max_entries', None) != int(max_entries):
        _cache = MoistureMonitorCache(max_entries=max_entries)
    return _cache


def append_sample_and_build_histories(mqtt_data: dict, mqtt_cfg: dict, now_ts: Optional[float] = None) -> Dict:
    """Append `mqtt_data` to cache and return `history` and `sensor_histories`.

    Returns: {'history': [...], 'sensor_histories': {...}}
    """
    # Optional asynchronous flow: if config enables `PUSH_ASYNC`, enqueue the
    # raw metric and return immediately. This keeps the MQTT/http input path
    # fast and non-blocking; separate worker processes should consume the
    # queue and perform the heavier meteorology and push operations.
    try:
        if mqtt_cfg and str(mqtt_cfg.get('PUSH_ASYNC', '')).lower() in ('1', 'true', 'yes'):
            try:
                from . import weather_station as metric_queue
                now = now_ts if now_ts is not None else __import__('time').time()
                payload = {'data': mqtt_data, '_ts': now}
                metric_queue.enqueue_metric(payload)
                return {'enqueued': True}
            except Exception:
                # Fall back to synchronous processing on queue failure.
                pass
    except Exception:
        pass
    max_entries = int(mqtt_cfg.get('PUSH_MAX_ENTRIES', 1000))
    cache = _ensure_cache(max_entries)
    now = now_ts if now_ts is not None else __import__('time').time()
    cache.append(now, mqtt_data)

    hours = float(mqtt_cfg.get('PUSH_HISTORY_HOURS', 3.0))
    seconds_window = int(hours * 3600)
    cache.prune(seconds_window, now_ts=now)
    history = cache.get_recent(seconds_window)

    # default mapping of canonical sensor names -> payload keys
    slot_map = mqtt_cfg.get('PUSH_SLOT_MAP') or {
        'rain': 'rain',
        'humidity': 'humidity',
        'pressure': 'pressure',
        'temperature': 'temperature',
        'dew_point': 'dew_point',
        'lightning': 'lightning',
    }
    packager = DataPackager(slot_map)
    # packager.build returns (raw, norm) mapping; maintain backwards
    # compatible keys and include normalized histories under
    # 'sensor_histories_norm'.
    built = packager.build(history, window_s=seconds_window, now_ts=now)
    if isinstance(built, tuple) or isinstance(built, list):
        sensor_histories = built[0]
        sensor_histories_norm = built[1] if len(built) > 1 else {}
    else:
        sensor_histories = built
        sensor_histories_norm = {}

    # --- Star-count based high-cloud heuristic ---
    # Use module-level defaults unless overridden in config.
    HIGH_CLOUD_STAR_THRESHOLD = int(mqtt_cfg.get('PUSH_HIGH_CLOUD_STAR_THRESHOLD', DEFAULT_HIGH_CLOUD_STAR_THRESHOLD))
    HIGH_CLOUD_AFTER_HOUR = int(mqtt_cfg.get('PUSH_HIGH_CLOUD_AFTER_HOUR', DEFAULT_HIGH_CLOUD_AFTER_HOUR))

    # Look for star count in the immediate mqtt payload under common keys.
    star_count = None
    for key in ('star_count', 'stars', 'stars_count', 'starCount'):
        if isinstance(mqtt_data, dict) and key in mqtt_data:
            try:
                star_count = int(mqtt_data[key])
                break
            except Exception:
                continue

    # fallback: if either raw or normalized histories contain a 'stars'
    # series, use its latest value. If only a normalized series is
    # available (0..1), scale it conservatively by the star threshold to
    # produce an integer-like estimate for downstream logic.
    if star_count is None:
        # prefer raw series when present
        s_raw = sensor_histories.get('stars') if isinstance(sensor_histories, dict) else None
        if s_raw:
            try:
                star_count = int(s_raw[-1][1])
            except Exception:
                star_count = None
        else:
            # try normalized series as a fallback
            s_norm = sensor_histories_norm.get('stars') if isinstance(sensor_histories_norm, dict) else None
            if s_norm:
                try:
                    # normalized value in [0,1] -> scale by threshold
                    star_count = int(float(s_norm[-1][1]) * float(HIGH_CLOUD_STAR_THRESHOLD))
                except Exception:
                    star_count = None

    # Determine local hour for the sample timestamp
    import time as _time
    sample_ts = int(now_ts if now_ts is not None else _time.time())
    local_hour = _time.localtime(sample_ts).tm_hour

    # Compute a graded `high_cloud_value` in [0.0, 1.0]. When star_count >=
    # threshold or before the configured hour, value is 0. If star_count is
    # 0, value is 1. Values scale linearly in between.
    high_cloud_value = None
    if star_count is not None and local_hour >= HIGH_CLOUD_AFTER_HOUR:
        try:
            sc = float(star_count)
            if sc >= float(HIGH_CLOUD_STAR_THRESHOLD):
                high_cloud_value = 0.0
            else:
                # clamp and invert
                ratio = max(0.0, min(1.0, (float(HIGH_CLOUD_STAR_THRESHOLD) - sc) / float(HIGH_CLOUD_STAR_THRESHOLD)))
                high_cloud_value = float(ratio)
        except Exception:
            high_cloud_value = None

    # Make the boolean decision explicit to avoid truthiness pitfalls
    # (e.g. 0.0 is falsy, None is falsy). Evaluate numeric comparison
    # safely and default to False on conversion errors.
    high_cloud = False
    if high_cloud_value is not None:
        try:
            high_cloud = float(high_cloud_value) > 0.0
        except Exception:
            high_cloud = False

    return {
        'history': history,
        'sensor_histories': sensor_histories,
        'sensor_histories_norm': sensor_histories_norm,
        'star_count': star_count,
        'high_cloud': high_cloud,
        'high_cloud_value': high_cloud_value,
        'high_cloud_threshold': HIGH_CLOUD_STAR_THRESHOLD,
        'high_cloud_after_hour': HIGH_CLOUD_AFTER_HOUR,
    }


def run_worker(mqtt_cfg: dict, shutdown_event: Optional['threading.Event'] = None):
    """Run loop suitable for a worker process: consume metrics, process, dispatch.

    Kept inside the Meteorologist module so maintainers find related logic
    together and naming/spelling stays consistent.
    """
    import threading
    import logging
    import traceback

    LOG = logging.getLogger('meteorologist.worker')

    if shutdown_event is None:
        shutdown_event = threading.Event()

    LOG.info('meteorologist worker: starting loop')
    while not shutdown_event.is_set():
        try:
            from . import weather_station as metric_queue
            metric = metric_queue.get_metric(block=True, timeout=1)
        except Exception:
            continue

        try:
            cfg = dict(mqtt_cfg or {})
            cfg['PUSH_ASYNC'] = False
            now_ts = None
            if isinstance(metric, dict) and metric.get('_ts'):
                now_ts = metric.get('_ts')
            data = metric.get('data') if isinstance(metric, dict) else metric
            result = append_sample_and_build_histories(data, cfg, now_ts=now_ts)
            # Optionally perform heavy analysis here so the dispatcher only
            # receives interpreted results. When enabled, the rain detection
            # model computes a score and a boolean `push_triggered` which are
            # attached to the dispatched result. This keeps the dispatcher
            # lightweight and focused on transport/gating.
            try:
                mqtt_cfg_local = cfg.get('MQTTPUBLISH', {})
                if mqtt_cfg_local.get('PUSH_USE_ANALYSIS', True):
                    from . import rain_detection_models as rdm
                    # Prefer normalized histories when available
                    histories_for_model = result.get('sensor_histories_norm') or result.get('sensor_histories', {})
                    score_obj = rdm.compute_weighted_rain_score_with_recent_priority(
                        histories_for_model,
                        long_window_s=int(float(mqtt_cfg_local.get('PUSH_LONG_WINDOW_S', 3 * 3600))),
                        short_window_s=int(float(mqtt_cfg_local.get('PUSH_SHORT_WINDOW_S', 30 * 60))),
                        spike_multiplier=float(mqtt_cfg_local.get('PUSH_SPIKE_MULTIPLIER', 2.0)),
                        absolute_delta=float(mqtt_cfg_local.get('PUSH_SPIKE_ABSOLUTE_DELTA', 0.5)),
                        # Prefer top-level `PUSH_SENSOR_WEIGHTS` if provided (cfg),
                        # otherwise fall back to `MQTTPUBLISH.PUSH_SENSOR_WEIGHTS`.
                        weights=(cfg.get('PUSH_SENSOR_WEIGHTS') or mqtt_cfg_local.get('PUSH_SENSOR_WEIGHTS') or None),
                        star_count=result.get('star_count'),
                        high_cloud=result.get('high_cloud_value') or result.get('high_cloud'),
                        high_cloud_multiplier=float(mqtt_cfg_local.get('PUSH_HIGH_CLOUD_MULTIPLIER', 1.2)),
                        slope_window_s=int(float(mqtt_cfg_local.get('PUSH_SLOPE_WINDOW_S', 30 * 60))),
                        hum_delta_norm=float(mqtt_cfg_local.get('PUSH_HUM_DELTA_NORM', 0.5)),
                        pres_delta_norm=float(mqtt_cfg_local.get('PUSH_PRES_DELTA_NORM', 1.0)),
                        hum_slope_weight=float(mqtt_cfg_local.get('PUSH_HUM_SLOPE_WEIGHT', 0.25)),
                        pres_slope_weight=float(mqtt_cfg_local.get('PUSH_PRES_SLOPE_WEIGHT', 0.25)),
                        slope_combined_multiplier=float(mqtt_cfg_local.get('PUSH_SLOPE_COMBINED_MULTIPLIER', 1.3)),
                        slope_signal_threshold=float(mqtt_cfg_local.get('PUSH_SLOPE_SIGNAL_THRESHOLD', 0.5)),
                    )
                    push_score = float(score_obj.get('score', 0.0))
                    result['push_score'] = push_score
                    threshold = float(mqtt_cfg_local.get('PUSH_SCORE_THRESHOLD', 0.0))
                    result['push_triggered'] = (push_score >= threshold)
            except Exception:
                LOG.exception('meteorologist worker: analysis failed')
            # --- Simple lightning distance/count trigger ---
            # If the user sets `PUSH_LIGHTNING_KM` (or count threshold),
            # evaluate it here and set `lightning_triggered` and
            # `push_triggered` (OR'd with existing decision). This keeps a
            # convenient, fast path for lightning-based alerts without
            # requiring users to author a full PUSH_MODEL expression.
            try:
                lightning_cfg = mqtt_cfg_local.get('PUSH_LIGHTNING_KM')
                if lightning_cfg is not None and str(lightning_cfg).strip() != '':
                    lk = float(lightning_cfg)
                    lightning_val = None
                    # Prefer immediate value from the incoming payload
                    try:
                        if isinstance(data, dict) and 'lightning' in data:
                            lightning_val = float(data.get('lightning'))
                    except Exception:
                        lightning_val = None
                    # Fallback to most recent sensor history value
                    if lightning_val is None:
                        try:
                            lh = result.get('sensor_histories', {}).get('lightning') or []
                            if lh:
                                lightning_val = float(lh[-1][1])
                        except Exception:
                            lightning_val = None
                    if lightning_val is not None:
                        lightning_triggered = (lightning_val <= lk)
                        result['lightning_value'] = lightning_val
                        result['lightning_triggered'] = bool(lightning_triggered)
                        if lightning_triggered:
                            result['push_triggered'] = True
            except Exception:
                LOG.exception('meteorologist worker: lightning trigger check failed')
            dispatch_item = {'result': result, 'raw': metric}
            metric_queue.enqueue_dispatch(dispatch_item)
        except Exception:
            LOG.error('meteorologist worker: failed processing metric: %s', traceback.format_exc())
    LOG.info('meteorologist worker: exiting')


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    run_worker({}, None)
