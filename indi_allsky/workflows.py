"""Workflow-oriented components: DataPackager, AnalysisWorkflow, PushMessaging.

Purpose and design
------------------
- `DataPackager` converts a runtime log-history (iterable of (ts,payload)
  tuples) into the `sensor_histories` mapping expected by the analysis
  module (`rain_detection_models`). This centralises how UI-configured
  sensor selections map to named inputs.
- `AnalysisWorkflow` is a thin wrapper that calls functions in
  `rain_detection_models` to produce scores/signals.
- `TimeGate` and `PushMessaging` provide a place to hold UI-configured
  time-based gating and the actual publishing logic. The time gate
  should be persisted/managed by the UI; here it's a configurable
  object that PushMessaging consults before sending.

These classes are intentionally small and dependency-free so the UI or a
unit-test can exercise them easily.
"""

from typing import Dict, Iterable, List, Optional
import time
import logging

from . import rain_detection_models as rdm
def get_series(history: Iterable, slot: str, window_s: Optional[float] = None, now_ts: Optional[float] = None):
    """Return a list of (ts, value) tuples for numeric values of `slot` in history.

    history is expected to be an iterable of (ts, payload) as produced by
    `PushHistoryCache.get_recent()` or similar. If `window_s` is provided the
    returned series will include only entries with ts >= now - window_s.
    """
    if now_ts is None:
        now_ts = __import__('time').time()
    cutoff_ts = now_ts - window_s if window_s else 0
    series = []
    for ts, d in history:
        if ts < cutoff_ts:
            continue
        if slot in d and isinstance(d[slot], (int, float)):
            series.append((ts, float(d[slot])))
    return series

logger = logging.getLogger('indi_allsky')


class DataPackager:
    """Builds the sensor_histories mapping from a log-history.

    - `history` is an iterable of (timestamp, payload) tuples (the same
      structure used elsewhere in the codebase).
    - `slot_map` maps canonical sensor names (e.g., 'humidity', 'rain')
      to the payload key/slot name present in the payload dict.
    - `window_s` optionally limits the returned series to recent seconds.
    """
    def __init__(self, slot_map: Dict[str, str]):
        self.slot_map = dict(slot_map)

    def build(self, history: Iterable, window_s: Optional[float] = None) -> Dict[str, List]:
        sensor_histories: Dict[str, List] = {}
        now_ts = None
        # get_series already can take a now_ts, but we'll let it decide
        for sensor, slot in self.slot_map.items():
            series = get_series(list(history), slot, window_s, now_ts)
            sensor_histories[sensor] = series
        return sensor_histories


class AnalysisWorkflow:
    """Wrapper around `rain_detection_models` for a simple analysis flow."""

    def __init__(self, analyzer_module=rdm):
        self.analyzer = analyzer_module

    def analyze(self, sensor_histories: Dict[str, Iterable], *,
                sensor_weights: Optional[Dict[str, float]] = None,
                include: Optional[Iterable[str]] = None,
                feature_key: str = rdm.DEFAULT_FEATURE_KEY,
                normalize: bool = True,
                **spike_kwargs) -> Dict:
        # prepare flattened input (useful for ML models)
        model_input = self.analyzer.prepare_model_input(sensor_histories)
        # compute weighted rain score with recent-priority spike detection
        score_obj = self.analyzer.compute_weighted_rain_score_with_recent_priority(
            sensor_histories,
            sensor_weights=sensor_weights,
            include=include,
            feature_key=feature_key,
            normalize=normalize,
            **spike_kwargs,
        )
        return {'model_input': model_input, 'score': score_obj}


class TimeGate:
    """Configurable, testable time gate used by `PushMessaging`.

    The UI should persist a configuration that this object enforces. The
    simplest configuration is a daily permit window defined by
    `start_hour`/`end_hour` in local time; more complex rules can be
    added later.
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
        # wrap-around (e.g., start=20, end=6)
        return h >= self.start_hour or h < self.end_hour


class PushMessaging:
    """Publish alerts subject to a `TimeGate`.

    `publisher` may be a callable (topic,payload) or an object with
    `publish(topic,payload)`.
    """

    def __init__(self, publisher, time_gate: Optional[TimeGate] = None):
        self.publisher = publisher
        self.time_gate = time_gate or TimeGate(enabled=False)

    def send(self, topic: str, payload: dict) -> None:
        if not self.time_gate.is_open():
            logger.info('Push suppressed by time gate')
            return
        try:
            if callable(self.publisher):
                self.publisher(topic, payload)
            elif hasattr(self.publisher, 'publish'):
                self.publisher.publish(topic, payload)
        except Exception:
            logger.exception('PushMessaging: error publishing')
