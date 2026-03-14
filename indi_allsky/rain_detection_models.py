"""Analysis primitives for rain detection.

This module provides pure, testable functions that accept per-sensor
histories (mapping sensor -> list of (ts, value) tuples) and produce
features, spike detections, and a weighted rain *probability-like* score.
State (history cache and packaging) is owned by `Meteorologist`.

Model overview (high level)
----------------------------
- Cloud cover: treated as a numeric degree (0.0..1.0) and applied as a
    multiplier to the final score; higher cloudiness increases the
    reported rain-likelihood. When enabled, cloudiness can be derived from
    star-count (fewer stars -> higher cloudiness) via an optional toggle in
    the Push settings.
- Lightning: treated as a boolean switch when configured (see
    `MQTTPUBLISH.PUSH_LIGHTNING_KM` in config). When enabled, callers
    (e.g., `Meteorologist`) should set a lightning presence flag if the
    sensor reports strikes within the configured distance; lightning is
    used as a determinant (boolean) rather than a numeric spike feature.
- Rain: exposed as a boolean presence indicator (recent short-window
    sensor value > 0). Rain is available both as a boolean determinant
    and as a numeric feature in the weighted sum (depending on config).
- Humidity: treated as a numeric feature (0..1) and used both in the weighted
    sum and for slope/spike boosting. Rapid rises in humidity can increase
    the rain likelihood score.
- Dew spread (relative humidity proxy): computed from temperature and dew point
    using standard vapor pressure relations, but only contributes when the dew point
    approaches the ambient temperature. The resulting 0..1 signal is used as a proxy
    for saturation and precipitation potential (raw dew point values are ignored unless
    the spread is small enough).
- Pressure tendency: implements standard weather tendency rules (e.g.,
    Truganina/Met Office) where a 1‑hour change of <0.5 hPa is considered
    "steady" and >0.5 hPa changes indicate rising/falling pressure. The
    benchmark is scaled linearly by the window length (e.g., 30m -> 0.25 hPa,
    3h -> 1.5 hPa) so shorter/longer windows still use the same reference.
- Trend & spike analysis: slope and spike detectors run primarily on
    `humidity` and `pressure` (short vs long windows). These signals are
    used to boost the score when rapid changes or spikes indicate a likely
    precipitation event.

Probability note
----------------
The output score is clamped to [0, 1] and is intended to behave like a
rain likelihood. The score may be further adjusted by spike boosts,
trend multipliers, and configurable high-cloud scaling.
"""
from typing import List, Tuple, Dict, Iterable, Optional
import statistics

# Include `high_cloud` as a first-class feature (not a physical sensor)
DEFAULT_SENSOR_WEIGHTS = {
    'rain': 1.0,
    'humidity': 0.5,
    'pressure': 0.2,
    # Temperature alone is not a reliable rain indicator; it is only
    # meaningful in the context of dew point (via dew_spread_inverse).
    # Therefore we do not weight raw temperature directly.
    'temperature': 0.0,
    # Raw dew_point is removed during processing; only the derived
    # dew_spread_inverse signal (based on temp/dew spread) contributes.
    'dew_point': 0.0,
    'dew_spread_inverse': 0.2,
    'lightning': 2.0,
    # Pressure tendency (rising/steady/falling) encoded as a 0..1 signal
    # based on standard weather practice (e.g., Truganina Weather / Met Office
    # pressure tendency codes).
    'pressure_tendency': 0.15,
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

    Defaults are tuned for the rain model:
      - humidity: triggers on ~15% jump in 30m (or ~25% over longer windows)
      - pressure: triggers on smaller changes (typical for pressure behavior)
    """
    if thresholds is None:
        thresholds = {
            # 15% change over ~30m should be considered a spike (shrinking from prior 0.8 default)
            'humidity': {'absolute_delta': 0.15, 'multiplier': 1.5},
            # pressure changes are smaller; keep a gentle threshold
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
    if sensor_histories is None:
        sensor_histories = {}
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
    # Clamp score to [0,1] so it behaves like a probability.
    score = max(0.0, min(1.0, float(score)))
    return {'score': float(score), 'breakdown': breakdown, 'total_weight': float(total_weight)}


def compute_weighted_rain_score_with_recent_priority(
    sensor_histories: Dict[str, List[Tuple[int, float]]],
    sensor_weights: Optional[Dict[str, float]] = None,
    include: Optional[Iterable[str]] = None,
    feature_key: str = DEFAULT_FEATURE_KEY,
    normalize: bool = True,
    # NOTE: For this function to return a meaningful probability-like score
    # in [0,1], callers should pass normalized sensor values (0..1). The
    # higher-level caller (`Meteorologist`) already prefers normalized
    # histories when available.
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
    # Pressure tendency rules: standard weather practice (e.g., Truganina
    # Weather's symbols and Met Office tendency codes) classify 1-hour pressure
    # change < 0.5 hPa as "steady", and >= 0.5 hPa as "rising" or "falling".
    # This implementation derives both short- and long-window tendency values
    # by scaling the 1-hour rule to the window length (30min -> 0.25 hPa, 3h ->
    # 1.5 hPa). This matches the intent of the published tendency code schemes
    # while allowing flexible window lengths.
    #
    # References:
    # - https://www.truganinaweather.com/weather-education/weather-symbols.htm
    # - https://www.mintakainnovations.com/wp-content/uploads/Pressure_Tendency_Characteristic_Code.pdf
    pressure_tendency_base_hpa: float = 0.5,
    # Slope-based humidity/pressure signals (see below) ---------------------------------
    # window (seconds) to consider for slope comparison (short window)
    slope_window_s: int = 30 * 60,
    # normalization deltas: delta across `slope_window_s` that maps -> 1.0
    # (15% change over 30 mins should be treated as a full signal)
    hum_delta_norm: float = 0.15,
    pres_delta_norm: float = 1.0,
    # relative weights for humidity/pressure slope signals when boosting score
    hum_slope_weight: float = 0.25,
    pres_slope_weight: float = 0.25,
    # when both signals are above a threshold, apply an extra combined multiplier
    slope_combined_multiplier: float = 1.3,
    slope_signal_threshold: float = 0.5,
    # humidity multiplier when humidity is high (e.g., 70%+)
    # This is a blending factor (0..1) that moves the final score closer to 1.0.
    high_humidity_threshold: float = 0.7,
    high_humidity_multiplier: float = 0.6,
    # Dew point spread signal (smaller spread -> higher rain likelihood)
    dew_spread_max: float = 6.0,
) -> Dict[str, object]:
    # build per-sensor long/short series
    details = {}
    long_histories = {}
    short_histories = {}

    # Allow missing/None sensor histories without failing.
    if sensor_histories is None:
        sensor_histories = {}
    # Use a consistent "now" across all sensors (latest known timestamp),
    # so windows are aligned and comparisons are stable.
    now = None
    for k, series in sensor_histories.items():
        if series:
            last_ts = series[-1][0]
            now = max(now, last_ts) if now is not None else last_ts
    if now is None:
        now = 0

    for k, series in sensor_histories.items():
        if series:
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

    # Determine pressure tendency based on change over the short/long windows.
    # This is modeled after standard weather tendency codes (Truganina/Met Office)
    # where <0.5 hPa change is steady, >=0.5 hPa is rising/falling.
    def _pressure_tendency(series: List[Tuple[int, float]], window_s: float) -> Optional[float]:
        if not series or len(series) < 2:
            return None
        try:
            t0, p0 = series[0]
            t1, p1 = series[-1]
            dt_hours = max(1e-6, (float(t1) - float(t0)) / 3600.0)
            dp = float(p1) - float(p0)
            rate = dp / dt_hours
            # Scale the 1-hour threshold to the window duration
            threshold = float(pressure_tendency_base_hpa) * (window_s / 3600.0)
            if abs(rate) < threshold:
                # steady
                return 0.5
            # rising -> 1.0, falling -> 0.0
            return 1.0 if rate > 0 else 0.0
        except Exception:
            return None

    pres_short_tendency = _pressure_tendency(short_histories.get('pressure', []) or [], short_window_s)
    pres_long_tendency = _pressure_tendency(long_histories.get('pressure', []) or [], long_window_s)
    pres_tendency = None
    if pres_short_tendency is not None and pres_long_tendency is not None:
        pres_tendency = (pres_short_tendency + pres_long_tendency) / 2.0
    elif pres_short_tendency is not None:
        pres_tendency = pres_short_tendency
    elif pres_long_tendency is not None:
        pres_tendency = pres_long_tendency

    if pres_tendency is not None:
        sensor_histories = dict(sensor_histories)
        sensor_histories['pressure_tendency'] = [(now, pres_tendency)]

    # Patched dew point handling: raw dew point is not used directly because a
    # high dew point alone (far below temperature) should not imply rain.
    # Instead we derive a single bounded signal that only becomes non-zero when
    # dew point approaches ambient temperature.
    dew_spread_signal = None
    temp_series = short_histories.get('temperature', []) or []
    dew_series = short_histories.get('dew_point', []) or []

    # Remove the raw dew_point sensor so it cannot contribute directly.
    sensor_histories = dict(sensor_histories)
    sensor_histories.pop('dew_point', None)

    if temp_series and dew_series:
        try:
            t = float(temp_series[-1][1])
            d = float(dew_series[-1][1])
            # Only contribute when dew point is near the ambient temperature.
            # If the spread is too large, treat it as "no contribution".
            spread = max(0.0, t - d)
            if spread <= float(dew_spread_max):
                # Use an RH-like computation for smooth scaling near saturation.
                a = 17.27
                b = 237.7
                sat_t = 6.112 * (2.718281828459045 ** ((a * t) / (b + t)))
                sat_d = 6.112 * (2.718281828459045 ** ((a * d) / (b + d)))
                rh = 0.0
                if sat_t > 0:
                    rh = max(0.0, min(1.0, sat_d / sat_t))
                # Scale the RH-like signal based on how close we are to saturation.
                dew_spread_signal = rh * (1.0 - (spread / float(dew_spread_max)))
        except Exception:
            dew_spread_signal = None

    # Add dew spread signal as a lightweight feature for the base score.
    # This lets weights (and user config) control its impact; it won't blow up
    # the score because it's bounded and treated like any other sensor.
    if dew_spread_signal is not None:
        sensor_histories['dew_spread_inverse'] = [(now, dew_spread_signal)]

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

    # Normalize to 0..1 using provided norms. The deltas are always numeric because
    # the helper returns 0.0 when data is missing, so the fallback branch is not
    # needed here.
    hum_sig = max(0.0, min(1.0, hum_delta / float(hum_delta_norm)))
    pres_sig = max(0.0, min(1.0, pres_delta / float(pres_delta_norm)))

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

    # Apply an additional boost when humidity is high (e.g., >70%).
    # Treat the score as a probability and avoid exceeding 1.0 by blending
    # toward 1.0 rather than multiplying (which can exceed 1.0).
    used_humidity_multiplier = None
    try:
        last_hum = None
        if short_histories.get('humidity'):
            last_hum = float(short_histories['humidity'][-1][1])
        if last_hum is not None and last_hum >= float(high_humidity_threshold):
            # Blend toward 1.0 instead of multiplying directly.
            # E.g. 0.8 + (1-0.8)*0.15 = 0.83, never > 1.0.
            used_humidity_multiplier = float(high_humidity_multiplier)
            score = float(score) + (1.0 - float(score)) * used_humidity_multiplier
    except Exception:
        used_humidity_multiplier = None

    # Clamp final score to <= 0.99 to maintain interpretations as a probability.
    score = min(float(score), 0.99)
    details['spike'] = {'humidity': hum_spike, 'pressure': pres_spike, 'any': spike_detected}
    details['rain_present'] = rain_present
    details['lightning_present'] = lightning_present
    details['dew_spread_inverse'] = float(dew_spread_signal) if dew_spread_signal is not None else None
    details['base'] = base
    details['star_count'] = int(star_count) if star_count is not None else None
    details['high_cloud'] = (high_cloud_value is not None and high_cloud_value > 0.0)
    details['high_cloud_value'] = float(high_cloud_value) if high_cloud_value is not None else None
    details['high_cloud_multiplier'] = float(used_multiplier) if used_multiplier is not None else None
    details['high_humidity_threshold'] = float(high_humidity_threshold)
    details['high_humidity_multiplier'] = float(used_humidity_multiplier) if used_humidity_multiplier is not None else None
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
