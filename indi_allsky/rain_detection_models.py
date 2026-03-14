"""Analysis primitives for rain detection.

This module provides pure, testable functions that accept per-sensor
histories (mapping sensor -> list of (ts, value) tuples) and produce
features, spike detections and a weighted rain score. State (history
cache and packaging) is owned by `Meteorologist`.

Model overview (high level)
----------------------------
- Cloud cover: treated as a numeric degree (0.0..1.0) and applied as a
    multiplier to the final score; higher cloudiness increases the
    reported rain-likelihood.
- Lightning: treated as a boolean switch when configured (see
    `MQTTPUBLISH.PUSH_LIGHTNING_KM` in config). When enabled, callers
    (e.g., `Meteorologist`) should set a lightning presence flag if the
    sensor reports strikes within the configured distance; lightning is
    used as a determinant (boolean) rather than a numeric spike feature.
- Rain: exposed as a boolean presence indicator (recent short-window
    sensor value > 0). Rain is available both as a boolean determinant
    and as a numeric feature in the weighted sum (depending on config).
- Trend & spike analysis: slope and spike detectors run primarily on
    `humidity` and `pressure` (short vs long windows). These signals
    are used to boost the score when rapid changes or spikes indicate a
    likely precipitation event.

Normalization note
------------------
Sensor magnitudes have different units/ranges (pressure ~1000 hPa,
humidity 0-100%, lightning 0-1). Callers or the module should normalize
per-sensor values into a comparable 0..1 range before applying weights
so large-magnitude sensors do not dominate the weighted sum.
"""
from typing import List, Tuple, Dict, Iterable, Optional
import statistics

# Include `high_cloud` as a first-class feature (not a physical sensor)
DEFAULT_SENSOR_WEIGHTS = {
    'rain': 1.0,
    'humidity': 0.5,
    'pressure': 0.2,
    'dew_point': 0.3,
    'lightning': 2.0,
    # weight for high_cloud: higher means high-cloud presence contributes more to
    # the computed rain score. Value is configurable via callers/config.
    'high_cloud': 0.4,
}
DEFAULT_FEATURE_KEY = 'overall_mean'


def linear_slope(series: List[Tuple[int, float]]) -> Optional[float]:
    if not series or len(series) < 2:
        return None
    n = len(series)
    xs = [float(s[0]) for s in series]
    ys = [float(s[1]) for s in series]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def moving_stats(series: List[Tuple[int, float]]) -> Dict[str, Optional[float]]:
    if not series:
        return {'min': None, 'max': None, 'mean': None, 'median': None}
    vals = [v for _, v in series]
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {'min': s[0], 'max': s[-1], 'mean': mean, 'median': median}


def detect_recent_spike(long_series: List[Tuple[int, float]], short_series: List[Tuple[int, float]],
                        absolute_delta: float = 0.5, multiplier: float = 2.0) -> bool:
    if not short_series or not long_series:
        return False
    long_stats = moving_stats(long_series)
    short_stats = moving_stats(short_series)
    if long_stats['median'] is None or short_stats['median'] is None:
        return False
    if short_stats['median'] - long_stats['median'] >= absolute_delta:
        return True
    if long_stats['median'] > 0 and (short_stats['median'] / (long_stats['median'] or 1.0)) >= multiplier:
        return True
    return False


def detect_recent_spikes_map(long_histories: Dict[str, List[Tuple[int, float]]],
                             short_histories: Dict[str, List[Tuple[int, float]]],
                             thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, bool]:
    """Detect recent spikes per-sensor using `detect_recent_spike`.

    thresholds: mapping sensor -> {'absolute_delta': float, 'multiplier': float}
    Returns a dict sensor->bool.
    """
    if thresholds is None:
        thresholds = {
            'humidity': {'absolute_delta': 0.8, 'multiplier': 1.5},
            'pressure': {'absolute_delta': 0.2, 'multiplier': 1.5},
        }
    result: Dict[str, bool] = {}
    for sensor, params in thresholds.items():
        try:
            abs_d = float(params.get('absolute_delta', 0.5))
            mult = float(params.get('multiplier', 2.0))
            result[sensor] = detect_recent_spike(long_histories.get(sensor, []), short_histories.get(sensor, []), absolute_delta=abs_d, multiplier=mult)
        except Exception:
            result[sensor] = False
    return result


def compute_weighted_rain_score(sensor_histories: Dict[str, List[Tuple[int, float]]],
                                sensor_weights: Optional[Dict[str, float]] = None,
                                include: Optional[Iterable[str]] = None,
                                feature_key: str = DEFAULT_FEATURE_KEY,
                                normalize: bool = True,
                                # numeric representation for high_cloud (0.0..1.0),
                                # or boolean which will be converted: True->1.0, False->0.0
                                high_cloud_value: Optional[float] = None) -> Dict[str, object]:
    breakdown: Dict[str, Dict[str, float]] = {}
    total_weight = 0.0
    weighted_sum = 0.0
    include_set = set(include) if include is not None else None
    weights = sensor_weights if sensor_weights is not None else DEFAULT_SENSOR_WEIGHTS

    for sensor, hist in sensor_histories.items():
        if include_set is not None and sensor not in include_set:
            continue
        weight = float(weights.get(sensor, 1.0))
        stats = moving_stats(hist)
        val = float(stats.get('mean') or 0.0) if feature_key == 'overall_mean' else float(stats.get('median') or 0.0)
        wval = weight * val
        breakdown[sensor] = {'value': val, 'weight': weight, 'weighted': wval}
        total_weight += weight
        weighted_sum += wval

    # Treat `high_cloud` as a feature if a numeric value was provided. This lets
    # callers (or Meteorologist) pass a high_cloud degree alongside sensor histories.
    if ('high_cloud' in weights) and (high_cloud_value is not None):
        if include_set is None or 'high_cloud' in include_set:
            hv = float(high_cloud_value) if not isinstance(high_cloud_value, bool) else (1.0 if high_cloud_value else 0.0)
            hweight = float(weights.get('high_cloud', 1.0))
            hwval = hweight * hv
            breakdown['high_cloud'] = {'value': hv, 'weight': hweight, 'weighted': hwval}
            total_weight += hweight
            weighted_sum += hwval

    score = weighted_sum / total_weight if (normalize and total_weight > 0) else weighted_sum
    return {'score': float(score), 'breakdown': breakdown, 'total_weight': float(total_weight)}


def compute_weighted_rain_score_with_recent_priority(
    sensor_histories: Dict[str, List[Tuple[int, float]]],
    sensor_weights: Optional[Dict[str, float]] = None,
    include: Optional[Iterable[str]] = None,
    feature_key: str = DEFAULT_FEATURE_KEY,
    normalize: bool = True,
    long_window_s: int = 3 * 3600,
    short_window_s: int = 30 * 60,
    spike_multiplier: float = 2.0,
    absolute_delta: float = 0.5,
    spike_weight_boost: float = 1.5,
    # optional star/cloud signals
    star_count: Optional[int] = None,
    high_cloud: Optional[bool] = None,
    # Multiplier applied when high_cloud is True. Default >1 boosts rain-likelihood.
    high_cloud_multiplier: float = 1.2,
    # Slope-based humidity/pressure signals (see below) ---------------------------------
    # window (seconds) to consider for slope comparison (short window)
    slope_window_s: int = 30 * 60,
    # normalization deltas: delta across `slope_window_s` that maps -> 1.0
    hum_delta_norm: float = 0.5,
    pres_delta_norm: float = 1.0,
    # relative weights for humidity/pressure slope signals when boosting score
    hum_slope_weight: float = 0.25,
    pres_slope_weight: float = 0.25,
    # when both signals are above a threshold, apply an extra combined multiplier
    slope_combined_multiplier: float = 1.3,
    slope_signal_threshold: float = 0.5,
) -> Dict[str, object]:
    # build per-sensor long/short series
    details = {}
    long_histories = {}
    short_histories = {}
    now = None
    for k, series in sensor_histories.items():
        if series:
            now = series[-1][0]
            long_cut = now - long_window_s
            short_cut = now - short_window_s
            long_histories[k] = [(t, v) for t, v in series if t >= long_cut]
            short_histories[k] = [(t, v) for t, v in series if t >= short_cut]
        else:
            long_histories[k] = []
            short_histories[k] = []

    # Spike detection: prefer humidity/pressure slope-based spikes. Treat rain
    # and lightning as boolean presence flags (determinants) rather than
    # primary spike detectors.
    per_sensor_spikes = detect_recent_spikes_map(long_histories, short_histories)
    hum_spike = bool(per_sensor_spikes.get('humidity', False))
    pres_spike = bool(per_sensor_spikes.get('pressure', False))
    spike_detected = bool(hum_spike or pres_spike)

    # Determine boolean presence for rain and lightning from short-window
    # recent values (fast indicator): present if latest short value > 0
    def _latest_present(series: List[Tuple[int, float]]) -> bool:
        try:
            if not series:
                return False
            return bool(float(series[-1][1]) > 0)
        except Exception:
            return False

    rain_present = _latest_present(short_histories.get('rain', []) or [])
    lightning_present = _latest_present(short_histories.get('lightning', []) or [])

    # derive a numeric high_cloud_value for inclusion in the weighted score
    # Note: `high_cloud` may be provided as a numeric degree in [0..1] or
    # as a boolean; numeric values are accepted and clamped, allowing callers
    # (e.g., `Meteorologist`) to pass a graded cloudiness signal.
    high_cloud_value = None
    if high_cloud is not None:
        # If the caller supplies a numeric degree (0..1), use it (clamped).
        if isinstance(high_cloud, (int, float)):
            try:
                hv = float(high_cloud)
                high_cloud_value = max(0.0, min(1.0, hv))
            except Exception:
                high_cloud_value = None
        else:
            # boolean-like -> numeric
            high_cloud_value = 1.0 if bool(high_cloud) else 0.0

    base = compute_weighted_rain_score(sensor_histories, sensor_weights, include, feature_key, normalize,
                                       high_cloud_value=high_cloud_value)
    score = float(base.get('score', 0.0))
    if spike_detected:
        score *= spike_weight_boost
    # --- Slope-based humidity & pressure boosting ---
    # Compute short vs long slopes for humidity and pressure. If the short
    # slope exceeds the long slope (for humidity) or is more negative than
    # the long slope (for pressure), derive a normalized signal in [0,1].
    try:
        hum_short_slope = linear_slope(short_histories.get('humidity', []) or [])
        hum_long_slope = linear_slope(long_histories.get('humidity', []) or [])
    except Exception:
        hum_short_slope = hum_long_slope = None
    try:
        pres_short_slope = linear_slope(short_histories.get('pressure', []) or [])
        pres_long_slope = linear_slope(long_histories.get('pressure', []) or [])
    except Exception:
        pres_short_slope = pres_long_slope = None

    def _delta_over_window(short_s, long_s):
        if short_s is None:
            return 0.0
        # Preserve negative long_s values (important for pressure behavior).
        # Only substitute 0.0 when long_s is actually None; do not use
        # truthiness which would collapse negative slopes to zero.
        long_s_val = 0.0 if long_s is None else float(long_s)
        # delta across window (slope * window) -> approximate change
        return float(short_s - long_s_val) * float(slope_window_s)

    hum_delta = max(0.0, _delta_over_window(hum_short_slope, hum_long_slope))
    pres_delta = max(0.0, -_delta_over_window(pres_short_slope, pres_long_slope))

    # normalize to 0..1 using provided norms
    hum_sig = max(0.0, min(1.0, hum_delta / float(hum_delta_norm))) if hum_delta is not None else 0.0
    pres_sig = max(0.0, min(1.0, pres_delta / float(pres_delta_norm))) if pres_delta is not None else 0.0

    # apply additive weighting to final score
    slope_boost = 1.0 + (hum_slope_weight * hum_sig) + (pres_slope_weight * pres_sig)
    # stronger combined effect if both signals exceed threshold
    if hum_sig >= float(slope_signal_threshold) and pres_sig >= float(slope_signal_threshold):
        slope_boost *= float(slope_combined_multiplier)
    try:
        score = float(score) * float(slope_boost)
    except Exception:
        score = float(score)
    # Apply optional high-cloud multiplier: if a positive `high_cloud_value` is
    # present, scale the final score. This is distinct from treating high_cloud
    # as a feature in the weighted sum — both are intentionally supported.
    used_multiplier = None
    if high_cloud_value is not None and high_cloud_value > 0.0:
        try:
            used_multiplier = float(high_cloud_multiplier)
            score = float(score) * used_multiplier
        except Exception:
            used_multiplier = 1.2
            score = float(score) * used_multiplier
    details['spike'] = {'humidity': hum_spike, 'pressure': pres_spike, 'any': spike_detected}
    details['rain_present'] = rain_present
    details['lightning_present'] = lightning_present
    details['base'] = base
    details['star_count'] = int(star_count) if star_count is not None else None
    details['high_cloud'] = (high_cloud_value is not None and high_cloud_value > 0.0)
    details['high_cloud_value'] = float(high_cloud_value) if high_cloud_value is not None else None
    details['high_cloud_multiplier'] = float(used_multiplier) if used_multiplier is not None else None
    details['slope'] = {
        'humidity': {'short_slope': hum_short_slope, 'long_slope': hum_long_slope, 'delta': hum_delta, 'signal': hum_sig},
        'pressure': {'short_slope': pres_short_slope, 'long_slope': pres_long_slope, 'delta': pres_delta, 'signal': pres_sig},
        'slope_boost': slope_boost,
    }
    details['long'] = {k: moving_stats(v) for k, v in long_histories.items()}
    details['short'] = {k: moving_stats(v) for k, v in short_histories.items()}
    return {'score': score, 'spike': spike_detected, 'details': details}


def available_feature_keys() -> List[str]:
    return list(DEFAULT_SENSOR_WEIGHTS.keys())


def analyze_past_sensor_data(history: Iterable, windows: Tuple[int, ...] = (3, 5, 15)) -> Dict[str, float]:
    values = []
    times = []
    for item in history:
        if isinstance(item, tuple) and len(item) >= 2:
            times.append(float(item[0]))
            values.append(float(item[1]))
        else:
            values.append(float(item))
    features: Dict[str, float] = {}
    if not values:
        return features
    features['overall_mean'] = float(statistics.mean(values))
    features['overall_median'] = float(statistics.median(values))
    features['recent_delta'] = float(values[-1] - values[-2]) if len(values) >= 2 else 0.0
    for w in windows:
        seg = values[-w:] if len(values) >= w else values
        features[f'last_{w}_mean'] = float(statistics.mean(seg)) if seg else 0.0
    return features


def prepare_model_input(sensor_histories: Dict[str, Iterable]) -> Dict[str, float]:
    model_input: Dict[str, float] = {}
    for sensor, hist in sensor_histories.items():
        feats = analyze_past_sensor_data(hist)
        for k, v in feats.items():
            model_input[f'{sensor}_{k}'] = float(v)
    return model_input
