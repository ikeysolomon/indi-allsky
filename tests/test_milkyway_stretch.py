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
