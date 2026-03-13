"""Meteorologist: history scanning, caching and packaging for analysis.

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

    def prune(self, seconds_window):
        import time
        if seconds_window <= 0:
            return
        cutoff = int(time.time()) - int(seconds_window)
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


class IndiAllSkyDateCalcs:
    """Minimal date calcs: returns current date and a 3-hour transition."""
    def __init__(self, config=None, position_av=None):
        self.config = config
        self.position_av = position_av

    def calcDayDate(self, now: datetime):
        return now.date()

    def getDayDate(self):
        return datetime.now().date()

    def getNextDayNightTransition(self):
        return datetime.now() + timedelta(hours=3)


class DataPackager:
    """Convert a log-history of (ts,payload) into per-sensor histories."""
    def __init__(self, slot_map: Dict[str, str]):
        self.slot_map = dict(slot_map)

    def build(self, history: Iterable, window_s: Optional[float] = None) -> Dict[str, List]:
        sensor_histories: Dict[str, List] = {}
        # Determine a reference 'now' timestamp from the most recent history
        # entry if available; fall back to wall-clock time. This lets callers
        # pass pre-assembled history and have windowing applied consistently.
        now_ts = None
        try:
            last = None
            for ts, _ in reversed(list(history)):
                last = ts
                break
            if last is not None:
                now_ts = int(last)
        except Exception:
            now_ts = None
        for sensor, slot in self.slot_map.items():
            series = []
            for ts, payload in list(history):
                if window_s is not None and now_ts is not None and ts < now_ts - window_s:
                    continue
                if isinstance(payload, dict) and slot in payload and isinstance(payload[slot], (int, float)):
                    series.append((ts, float(payload[slot])))
            sensor_histories[sensor] = series
        return sensor_histories


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
    max_entries = int(mqtt_cfg.get('PUSH_MAX_ENTRIES', 1000))
    cache = _ensure_cache(max_entries)

    now = now_ts if now_ts is not None else __import__('time').time()
    cache.append(now, mqtt_data)

    hours = float(mqtt_cfg.get('PUSH_HISTORY_HOURS', 3.0))
    seconds_window = int(hours * 3600)
    cache.prune(seconds_window)
    history = cache.get_recent(seconds_window)

    # default mapping of canonical sensor names -> payload keys
    slot_map = mqtt_cfg.get('PUSH_SLOT_MAP') or {
        'rain': 'rain',
        'humidity': 'humidity',
        'pressure': 'pressure',
        'dew_point': 'dew_point',
        'lightning': 'lightning',
    }
    packager = DataPackager(slot_map)
    sensor_histories = packager.build(history, window_s=seconds_window)

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

    # fallback: if sensor_histories contains a 'stars' series, use its latest value
    if star_count is None and 'stars' in sensor_histories:
        s = sensor_histories.get('stars') or []
        if s:
            try:
                star_count = int(s[-1][1])
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

    high_cloud = bool(high_cloud_value and high_cloud_value > 0.0)

    return {
        'history': history,
        'sensor_histories': sensor_histories,
        'star_count': star_count,
        'high_cloud': high_cloud,
        'high_cloud_value': high_cloud_value,
        'high_cloud_threshold': HIGH_CLOUD_STAR_THRESHOLD,
        'high_cloud_after_hour': HIGH_CLOUD_AFTER_HOUR,
    }
