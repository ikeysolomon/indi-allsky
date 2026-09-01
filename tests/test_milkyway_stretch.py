import numpy
import pytest

from indi_allsky.milkyway import IndiAllskyMilkyWayStretch
from indi_allsky.milkyway import _GALACTIC_PLANE_CATALOG


def test_galactic_plane_catalog_matches_known_galactic_center():
    # l=0, b=0 (galactic center) is the 181st sampled longitude (-180..180
    # step 1) and has well known equatorial coordinates; this guards the
    # equatorial<->galactic rotation matrix/multiplication order against
    # silent regressions (e.g. wrong transpose or a non-orthogonal matrix).
    ra, dec = _GALACTIC_PLANE_CATALOG[180]
    assert ra == pytest.approx(266.405, abs=0.05)
    assert dec == pytest.approx(-28.936, abs=0.05)


def _config(enabled=True):
    return {
        'LENS_AZIMUTH': 0.0,
        'VIRTUALSKY': {
            'IMAGE_CIRCLE_DIAMETER': 1700,
            'LATITUDE_OFFSET': 0.0,
            'LONGITUDE_OFFSET': 0.0,
            'OFFSET_X': 0,
            'OFFSET_Y': 0,
        },
        'IMAGE_STRETCH': {
            'MILKYWAY_ENABLE': enabled,
            'MILKYWAY_MOONMODE': False,
            'MILKYWAY_GAMMA': 1.35,
            'MILKYWAY_BAND_WIDTH': 14.0,
            'MILKYWAY_FEATHER': 80.0,
        },
    }


def test_milkyway_stretch_is_opt_in_and_localized():
    image = numpy.full((720, 1280, 3), 40, dtype=numpy.uint8)
    disabled = IndiAllskyMilkyWayStretch(_config(enabled=False))
    assert disabled.apply(image, 45.0, -93.0, 1767225600.0) is image

    enabled = IndiAllskyMilkyWayStretch(_config())
    result = enabled.apply(image, 45.0, -93.0, 1767225600.0)

    assert numpy.any(result != image)
    assert numpy.any(result == image)
    assert enabled.last_elapsed_ms > 0.0


def test_milkyway_stretch_is_disabled_during_moonmode_by_default():
    image = numpy.full((720, 1280, 3), 40, dtype=numpy.uint8)
    enhancer = IndiAllskyMilkyWayStretch(_config())

    assert enhancer.apply(image, 45.0, -93.0, 1767225600.0, moonmode=True) is image
    assert numpy.any(enhancer.apply(image, 45.0, -93.0, 1767225600.0, moonmode=False) != image)


def test_milkyway_stretch_moonmode_toggle_allows_it_through():
    image = numpy.full((720, 1280, 3), 40, dtype=numpy.uint8)
    config = _config()
    config['IMAGE_STRETCH']['MILKYWAY_MOONMODE'] = True
    enhancer = IndiAllskyMilkyWayStretch(config)

    result = enhancer.apply(image, 45.0, -93.0, 1767225600.0, moonmode=True)
    assert numpy.any(result != image)


def test_milkyway_band_tracks_sidereal_time():
    # the whole point of piggybacking on the lens solution is that the band
    # follows the sky; sampling six hours apart must move it, not just
    # change it by coincidence of noise.
    image = numpy.full((1080, 1920, 3), 40, dtype=numpy.uint8)
    enhancer = IndiAllskyMilkyWayStretch(_config())

    result_a = enhancer.apply(image, 45.0, -93.0, 1767225600.0)
    result_b = enhancer.apply(image, 45.0, -93.0, 1767225600.0 + 6 * 3600)

    assert not numpy.array_equal(result_a, result_b)


def test_milkyway_gamma_brightens_the_masked_region_monotonically():
    image = numpy.full((1080, 1920, 3), 40, dtype=numpy.uint8)

    config_mild = _config()
    config_mild['IMAGE_STRETCH']['MILKYWAY_GAMMA'] = 1.1
    config_strong = _config()
    config_strong['IMAGE_STRETCH']['MILKYWAY_GAMMA'] = 3.0

    mild = IndiAllskyMilkyWayStretch(config_mild).apply(image, 45.0, -93.0, 1767225600.0)
    strong = IndiAllskyMilkyWayStretch(config_strong).apply(image, 45.0, -93.0, 1767225600.0)

    assert strong.astype(numpy.int32).sum() > mild.astype(numpy.int32).sum() > image.astype(numpy.int32).sum()


def test_milkyway_stretch_respects_binning():
    # binned frames are smaller and use a proportionally smaller image
    # circle; this must not raise or silently no-op.
    image = numpy.full((540, 960, 3), 40, dtype=numpy.uint8)
    enhancer = IndiAllskyMilkyWayStretch(_config())
    result = enhancer.apply(image, 45.0, -93.0, 1767225600.0, binning=2)
    assert numpy.any(result != image)


def _run_and_time(enhancer, image):
    enhancer.apply(image, 45.0, -93.0, 1767225600.0)
    return enhancer.last_elapsed_ms


def test_milkyway_stretch_stays_within_performance_budget():
    # explicit product requirement: no more than ~100ms per image, even at
    # high resolution. OpenCV pays a one-time per-process init cost on the
    # first call to distanceTransform/resize/LUT/blendLinear; the real
    # process is a long-lived capture daemon that only ever pays that once,
    # so warm up before measuring steady-state cost. Best-of-N further
    # filters out incidental OS scheduler/GC noise.
    image = numpy.full((2160, 3840, 3), 40, dtype=numpy.uint8)
    enhancer = IndiAllskyMilkyWayStretch(_config())
    _run_and_time(enhancer, image)  # warm-up, discarded
    best_ms = min(_run_and_time(enhancer, image) for _ in range(5))
    assert best_ms < 100.0


def test_milkyway_stretch_never_raises_on_bad_config():
    image = numpy.full((720, 1280, 3), 40, dtype=numpy.uint8)
    config = _config()
    config['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'] = 'not-a-number'
    enhancer = IndiAllskyMilkyWayStretch(config)
    assert enhancer.apply(image, 45.0, -93.0, 1767225600.0) is image
