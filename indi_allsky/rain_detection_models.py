"""
rain_detection_models.py

Analysis workflow role
----------------------
This module contains the analysis primitives and small feature-extraction
pipeline used by the rain-detection workflow. It is intentionally
focused on transforming per-sensor histories (sequences of numeric
values or `(timestamp, value)` pairs) into model-ready features and
signals. Typical workflow:

- `DataPackager` (a workflow component, planned in
    `indi_allsky/workflows.py`) will build a mapping of canonical sensor
    names to histories (e.g. `{'humidity': [(ts,v), ...], 'rain': [...], ...}`)
    by reading the log history (the same tuples that `PushEvaluator` uses).
- `AnalysisWorkflow` calls into this module with that `sensor_histories`
    mapping. `prepare_model_input()` flattens per-sensor features; the
    various `compute_*` helpers produce decision signals and weighted
    scores.

Where log history maps to analysis inputs:
- The canonical conversion is `history (iterable of (ts,payload))`
    -> `get_series()` (in `workflows.py` / `DataPackager`) -> per-sensor `(ts, value)` lists.
    `DataPackager` is responsible for calling `get_series()` for each
    configured sensor slot and providing the result to the functions in
    this module. See `prepare_model_input()` for the expected mapping
    shape: `Dict[str, Iterable]` where each value is either a list of
    values or a list of `(ts, value)` pairs.

Design intent:
- Keep analysis pure and dependency-free. This module should not
    access global runtime arrays or config; it accepts explicit data
    structures so it can be tested in isolation and reused across
    multiple workflows (MQTT push, internal alerting, web UI signals).
"""
from typing import List, Tuple, Dict, Iterable, Optional
import math
import statistics
import time

# Module-level constants (hoisted defaults)
# Default window sizes (in number of samples) used for computing windowed stats
DEFAULT_WINDOWS: Tuple[int, ...] = (3, 5, 15)
# Threshold used to classify slope as increasing/decreasing/stable
DEFAULT_POS_THRESHOLD: float = 1e-6
# Default feature to use when computing weighted rain score
DEFAULT_FEATURE_KEY: str = 'overall_mean'

# Supported sensor keys (common environmental sensors used in rain detection)
DEFAULT_SUPPORTED_SENSORS: Tuple[str, ...] = ('humidity', 'lightning', 'dew_point', 'pressure', 'rain')

# Default per-sensor weights (used when no explicit weights provided)
# Explicitly list each supported sensor with default weight 1.0
DEFAULT_SENSOR_WEIGHTS: Dict[str, float] = {
    'humidity': 1.0,
    'lightning': 1.0,
    'dew_point': 1.0,
    'pressure': 1.0,
    'rain': 1.0,
}

def _unpack_history(history: Iterable) -> Tuple[List[float], Optional[List[float]]]:
    """
    Normalize history input into (values, times).

    Accepts either:
    - a sequence of numbers [v0, v1, ...]
    - a sequence of (timestamp, value) pairs [(t0, v0), (t1, v1), ...]

    Returns (values, times) where times is None if not provided.
    """
    history = list(history)
    if not history:
        return [], None
    first = history[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        times = [float(t) for t, _ in history]
        values = [float(v) for _, v in history]
        return values, times
    else:
        values = [float(v) for v in history]
        return values, None


def linear_slope(values: List[float], times: Optional[List[float]] = None) -> float:
    """
    Compute slope of `values` over `times` using simple linear regression
    (least-squares). If `times` is None, uses integer indices as x.

    Returns slope (change in value per unit time/index). Returns 0.0
    for empty or constant-series inputs.
    """
    if not values:
        return 0.0
    n = len(values)
    if times is None:
        x = list(range(n))
    else:
        x = list(times)
    if len(x) != n:
        raise ValueError("times and values must have same length")
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(values)
    num = 0.0
    den = 0.0
    for xi, yi in zip(x, values):
        dx = xi - mean_x
        num += dx * (yi - mean_y)
        den += dx * dx
    if den == 0.0:
        return 0.0
    return num / den


def slope_trend(values: List[float], times: Optional[List[float]] = None, *,
                pos_threshold: float = DEFAULT_POS_THRESHOLD, neg_threshold: Optional[float] = None) -> str:
    """
    Classify the trend of `values` as 'increasing', 'decreasing', or 'stable'.

    Thresholds control sensitivity; by default a very small threshold is
    used (suitable for raw sensor units). Set `neg_threshold` to override
    the negative threshold; otherwise it is `-pos_threshold`.
    """
    if neg_threshold is None:
        neg_threshold = -pos_threshold
    s = linear_slope(values, times)
    if s > pos_threshold:
        return 'increasing'
    if s < neg_threshold:
        return 'decreasing'
    return 'stable'


def moving_stats(values: List[float]) -> Dict[str, float]:
    """Return basic statistics for a sequence of values."""
    if not values:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    mean = statistics.mean(values)
    try:
        std = statistics.pstdev(values)
    except Exception:
        std = 0.0
    return {
        'mean': mean,
        'median': statistics.median(values),
        'std': std,
        'min': min(values),
        'max': max(values),
    }


def analyze_past_sensor_data(history: Iterable, windows: Tuple[int, ...] = DEFAULT_WINDOWS) -> Dict[str, float]:
    """
    Given a sensor history (values or (time, value) pairs), compute
    a set of descriptive features suitable for model input or rules.

    Features produced include moving stats over multiple windows, the
    overall slope, recent delta, and counts of positive/negative changes.

    Returns a flat dict of feature_name -> value.
    """
    values, times = _unpack_history(history)
    features: Dict[str, float] = {}
    n = len(values)
    # overall stats
    features.update({f'overall_{k}': v for k, v in moving_stats(values).items()})
    features['overall_slope'] = linear_slope(values, times)
    # recent delta (last value minus value before that)
    if n >= 2:
        features['recent_delta'] = values[-1] - values[-2]
    else:
        features['recent_delta'] = 0.0
    # changes count
    pos = 0
    neg = 0
    for a, b in zip(values, values[1:]):
        if b > a:
            pos += 1
        elif b < a:
            neg += 1
    features['count_increases'] = float(pos)
    features['count_decreases'] = float(neg)

    # windowed stats
    for w in windows:
        if w <= 0:
            continue
        key = f'last_{w}'
        segment = values[-w:] if n >= w else values
        stats = moving_stats(segment)
        features.update({f'{key}_{k}': v for k, v in stats.items()})
        features[f'{key}_slope'] = linear_slope(segment, None if times is None else times[-len(segment):])

    return features


def prepare_model_input(sensor_histories: Dict[str, Iterable]) -> Dict[str, float]:
    """
    Given a mapping of sensor name -> history, produce a flattened feature
    dictionary combining features for each sensor. The output keys are
    prefixed by the sensor name (e.g. `rain_mean`, `humidity_last_5_std`).

    This function centralizes which sensors and features are passed to
    downstream models and can be extended as model needs evolve.
    """
    model_input: Dict[str, float] = {}
    for sensor, hist in sensor_histories.items():
        feats = analyze_past_sensor_data(hist)
        for k, v in feats.items():
            model_input[f'{sensor}_{k}'] = float(v)
    return model_input


def compute_weighted_rain_score(
    sensor_histories: Dict[str, Iterable],
    sensor_weights: Optional[Dict[str, float]] = None,
    include: Optional[Iterable[str]] = None,
    feature_key: str = DEFAULT_FEATURE_KEY,
    normalize: bool = True,
) -> Dict[str, object]:
    """
    Compute a weighted rain score from multiple sensor histories.

    - `sensor_histories`: mapping sensor name -> history (values or (time,value)).
    - `sensor_weights`: optional mapping sensor -> weight. Missing sensors get weight 1.0.
    - `include`: optional iterable of sensor names to include; others are ignored.
    - `feature_key`: which feature from `analyze_past_sensor_data` to use (e.g. 'overall_mean', 'last_5_mean', 'recent_delta').
    - `normalize`: if True, divides weighted sum by total weight to produce weighted average.

    Returns a dict with numeric `score` and a `breakdown` per sensor.
    """
    breakdown: Dict[str, Dict[str, float]] = {}
    total_weight = 0.0
    weighted_sum = 0.0
    include_set = set(include) if include is not None else None

    # use default weights when none provided
    weights = sensor_weights if sensor_weights is not None else DEFAULT_SENSOR_WEIGHTS

    for sensor, hist in sensor_histories.items():
        if include_set is not None and sensor not in include_set:
            continue
        weight = float(weights.get(sensor, 1.0))
        feats = analyze_past_sensor_data(hist)
        val = float(feats.get(feature_key, 0.0))
        wval = weight * val
        breakdown[sensor] = {'value': val, 'weight': weight, 'weighted': wval}
        total_weight += weight
        weighted_sum += wval

    score = weighted_sum / total_weight if (normalize and total_weight > 0) else weighted_sum
    return {'score': float(score), 'breakdown': breakdown, 'total_weight': float(total_weight)}


def detect_recent_spike(
    history: Iterable,
    now_ts: Optional[float] = None,
    long_window_s: int = 3 * 3600,
    short_window_s: int = 30 * 60,
    spike_multiplier: float = 2.0,
    absolute_delta: Optional[float] = None,
) -> Dict[str, object]:
    """
    Detect whether recent readings spike compared to a longer-term window.

    - `history`: iterable of values or (timestamp, value) pairs (timestamps expected as seconds since epoch).
    - `now_ts`: reference timestamp (defaults to last timestamp in history or current time).
    - `long_window_s`, `short_window_s`: windows in seconds.
    - `spike_multiplier`: short_mean > long_mean * spike_multiplier flags a spike.
    - `absolute_delta`: optional absolute difference threshold: short_mean - long_mean > absolute_delta also flags spike.

    Returns a dict with `short_mean`, `long_mean`, `delta`, `ratio`, and boolean `is_spike`.
    """
    values, times = _unpack_history(history)
    if not values:
        return {'short_mean': 0.0, 'long_mean': 0.0, 'delta': 0.0, 'ratio': 0.0, 'is_spike': False}

    # prefer timestamped histories for precise windows
    if times:
        now = float(now_ts) if now_ts is not None else float(times[-1])
        short_vals = [v for t, v in zip(times, values) if t >= now - short_window_s]
        long_vals = [v for t, v in zip(times, values) if t >= now - long_window_s]
    else:
        # fallback: treat windows as sample counts (short=3, long= max(15, short*5))
        short_count = max(1, min(len(values), 3))
        long_count = max(short_count, min(len(values), 15))
        short_vals = values[-short_count:]
        long_vals = values[-long_count:]

    def mean_or_zero(seq: List[float]) -> float:
        try:
            return float(statistics.mean(seq)) if seq else 0.0
        except Exception:
            return 0.0

    short_mean = mean_or_zero(short_vals)
    long_mean = mean_or_zero(long_vals)
    delta = short_mean - long_mean
    ratio = (short_mean / long_mean) if (long_mean and long_mean != 0.0) else (float('inf') if short_mean else 0.0)

    is_spike = False
    if long_mean and short_mean > long_mean * float(spike_multiplier):
        is_spike = True
    if absolute_delta is not None and delta > float(absolute_delta):
        is_spike = True

    return {
        'short_mean': short_mean,
        'long_mean': long_mean,
        'delta': delta,
        'ratio': ratio,
        'is_spike': bool(is_spike),
    }


def compute_weighted_rain_score_with_recent_priority(
    sensor_histories: Dict[str, Iterable],
    sensor_weights: Optional[Dict[str, float]] = None,
    include: Optional[Iterable[str]] = None,
    feature_key: str = DEFAULT_FEATURE_KEY,
    normalize: bool = True,
    long_window_s: int = 3 * 3600,
    short_window_s: int = 30 * 60,
    spike_multiplier: float = 2.0,
    absolute_delta: Optional[float] = None,
    spike_weight_boost: float = 2.0,
) -> Dict[str, object]:
    """
    Like `compute_weighted_rain_score` but detects recent spikes per sensor
    and increases their effective weight when a spike is present.

    Returns the same dict as `compute_weighted_rain_score` with additional
    `spike_info` per sensor included in the breakdown.
    """
    breakdown: Dict[str, Dict[str, float]] = {}
    total_weight = 0.0
    weighted_sum = 0.0
    include_set = set(include) if include is not None else None

    # use default weights when none provided
    weights = sensor_weights if sensor_weights is not None else DEFAULT_SENSOR_WEIGHTS

    for sensor, hist in sensor_histories.items():
        if include_set is not None and sensor not in include_set:
            continue
        base_weight = float(weights.get(sensor, 1.0))

        # detect spike using timestamps when available
        spike_info = detect_recent_spike(hist, long_window_s=long_window_s, short_window_s=short_window_s,
                                        spike_multiplier=spike_multiplier, absolute_delta=absolute_delta)

        effective_weight = base_weight * (float(spike_weight_boost) if spike_info.get('is_spike') else 1.0)

        feats = analyze_past_sensor_data(hist)
        val = float(feats.get(feature_key, 0.0))
        wval = effective_weight * val
        breakdown[sensor] = {
            'value': val,
            'base_weight': base_weight,
            'effective_weight': effective_weight,
            'weighted': wval,
            'spike_info': spike_info,
        }
        total_weight += effective_weight
        weighted_sum += wval

    score = weighted_sum / total_weight if (normalize and total_weight > 0) else weighted_sum
    return {'score': float(score), 'breakdown': breakdown, 'total_weight': float(total_weight)}


def available_feature_keys() -> List[str]:
    """Return a list of common feature keys produced by analyze_past_sensor_data."""
    # This reflects the keys used by analyze_past_sensor_data by default.
    keys = [
        'overall_mean', 'overall_median', 'overall_std', 'overall_min', 'overall_max', 'overall_slope',
        'recent_delta', 'count_increases', 'count_decreases'
    ]
    for w in DEFAULT_WINDOWS:
        keys.extend([f'last_{w}_mean', f'last_{w}_median', f'last_{w}_std', f'last_{w}_min', f'last_{w}_max', f'last_{w}_slope'])
    return keys


__all__ = [
    'linear_slope',
    'slope_trend',
    'analyze_past_sensor_data',
    'prepare_model_input',
    'compute_weighted_rain_score',
    'detect_recent_spike',
    'compute_weighted_rain_score_with_recent_priority',
    'available_feature_keys',
    'trend_slope',
    'increasing',
    'decreasing',
]

def trend_slope(history: Iterable, slot: str, window_s: Optional[float] = None) -> float:
    """Estimate trend slope (per minute) for a numeric `slot` in a push-style history.

    `history` is an iterable of (ts, payload) tuples. This helper extracts
    the numeric series for `slot` and computes slope using `linear_slope`.
    Returns slope in units per minute. If insufficient data, returns 0.0.
    """
    # extract series
    series = []
    for item in history:
        if not item:
            continue
        try:
            ts, payload = item
        except Exception:
            continue
        if isinstance(payload, dict) and slot in payload and isinstance(payload[slot], (int, float)):
            series.append((float(ts), float(payload[slot])))
    if len(series) < 2:
        return 0.0
    xs = [float(t) for t, v in series]
    ys = [float(v) for t, v in series]
    base = xs[0]
    offset_times = [x - base for x in xs]
    try:
        slope_per_second = linear_slope(ys, offset_times)
        return slope_per_second * 60.0
    except Exception:
        return 0.0


def increasing(history: Iterable, slot: str, window_s: Optional[float] = None, min_slope: float = 0.0) -> bool:
    return trend_slope(history, slot, window_s) > float(min_slope)


def decreasing(history: Iterable, slot: str, window_s: Optional[float] = None, min_slope: float = 0.0) -> bool:
    return trend_slope(history, slot, window_s) < -float(min_slope)
