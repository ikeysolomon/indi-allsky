import cv2
import numpy
import pytest

from indi_allsky.lens_solver.projection import predictAltAz
from indi_allsky.lens_solver.projection import projectToPixels
from indi_allsky.milkyway import IndiAllskyMilkyWayStretch
from indi_allsky.milkyway import _GALACTIC_PLANE_CATALOG
from indi_allsky.milkyway import base_stretch_allowed


def test_galactic_plane_catalog_matches_known_galactic_center():
    # l=0, b=0 (galactic center) is the 181st sampled longitude (-180..180
    # step 1) and has well known equatorial coordinates; this guards the
    # equatorial<->galactic rotation matrix/multiplication order against
    # silent regressions (e.g. wrong transpose or a non-orthogonal matrix).
    ra, dec = _GALACTIC_PLANE_CATALOG[180]
    assert ra == pytest.approx(266.405, abs=0.05)
    assert dec == pytest.approx(-28.936, abs=0.05)


def test_galactic_plane_catalog_matches_known_anticenter():
    # l=180, b=0 (galactic anticenter) has well known equatorial coordinates
    # (~RA 85.5deg, Dec +28.9deg); this is the catalog's other pole from the
    # galactic center and guards the same rotation matrix from a different
    # angle (literally) of the sky.
    ra, dec = _GALACTIC_PLANE_CATALOG[0]
    assert ra == pytest.approx(86.405, abs=0.5)
    assert dec == pytest.approx(28.936, abs=0.05)


def test_galactic_plane_catalog_antipodal_longitudes_are_consistent():
    # for any l, the point at l+180 must be antipodal on the galactic
    # equator: RA differs by exactly 180deg and Dec has the opposite sign
    # with the same magnitude. This is a structural invariant of the
    # rotation (independent of any external reference value), so it catches
    # a wrong transpose/sign regression at *every* sampled longitude, not
    # just the two named ones above.
    for l in (30, 60, 90, 120, 150):
        ra_a, dec_a = _GALACTIC_PLANE_CATALOG[l + 180]
        ra_b, dec_b = _GALACTIC_PLANE_CATALOG[l]  # l - 180, i.e. the antipode
        assert (ra_a - ra_b) % 360.0 == pytest.approx(180.0, abs=0.05)
        assert dec_a == pytest.approx(-dec_b, abs=0.05)


def test_galactic_plane_partially_below_horizon_for_typical_frame():
    # a typical mid-latitude frame must show *some* of the plane below the
    # horizon and *some* above it -- if every point were visible (or none
    # were), the alt >= -2deg cutoff used by _apply's visibility mask would
    # not actually be filtering anything, silently.
    alt, _ = predictAltAz(_GALACTIC_PLANE_CATALOG, 45.0, -93.0, 1767225600.0)
    visible = alt >= numpy.radians(-2.0)
    assert visible.any()
    assert not visible.all()


def test_predict_alt_az_hemisphere_sign_convention_is_consistent_across_a_day():
    # a star 5deg from the north celestial pole must be circumpolar (always
    # above the horizon) from a matching-sign (northern) latitude, and never
    # rise from the opposite-sign (southern) latitude of the same
    # magnitude, at every hour of the day. A sign error in the
    # latitude/declination handling would flip this for at least some hours.
    catalog = numpy.array([[0.0, 85.0]])
    base_obstime = 1767225600.0
    for hours in range(0, 24, 3):
        obstime = base_obstime + hours * 3600.0
        alt_north, _ = predictAltAz(catalog, 50.0, 0.0, obstime)
        alt_south, _ = predictAltAz(catalog, -50.0, 0.0, obstime)
        assert alt_north[0] > 0.0
        assert alt_south[0] < 0.0


def _config(enabled=True):
    return {
        'LENS_AZIMUTH': 0.0,
        'LENS_SOLVED': True,
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
            'MILKYWAY_GAMMA': 2.2,
            'MILKYWAY_BAND_WIDTH': 10.0,
            'MILKYWAY_FEATHER': 60.0,
            'MILKYWAY_SATURATION': 1.4,
        },
    }


def test_rendered_pixel_matches_shared_projection_end_to_end():
    # end-to-end regression anchor: the earlier tests prove the catalog and
    # the alt/az math are each internally consistent, but not that the full
    # chain -- Galactic catalog -> predictAltAz -> projectToPixels -> pixels
    # actually blended into the frame -- lands where the shared projection
    # module independently says it should. Galactic Centre is well above
    # the horizon at this lat/lon/time (unlike -93deg longitude used
    # elsewhere in this file, where it's below the horizon).
    lat, lon, obstime = -27.0, 153.0, 1767225600.0
    config = _config()
    image = numpy.full((1080, 1920, 3), 40, dtype=numpy.uint8)

    galactic_center = numpy.array([[266.405, -28.936]])
    alt, az = predictAltAz(galactic_center, lat, lon, obstime)
    params = (
        config['LENS_AZIMUTH'],
        config['VIRTUALSKY']['LATITUDE_OFFSET'],
        config['VIRTUALSKY']['LONGITUDE_OFFSET'],
        config['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'],
        config['VIRTUALSKY']['OFFSET_X'],
        config['VIRTUALSKY']['OFFSET_Y'],
    )
    expected_x, expected_y = projectToPixels(alt, az, params, 1920, 1080)
    ex, ey = int(round(expected_x[0])), int(round(expected_y[0]))
    assert 0 <= ex < 1920 and 0 <= ey < 1080  # sanity: must land on-frame

    result = IndiAllskyMilkyWayStretch(config).apply(image, lat, lon, obstime)

    assert result[ey, ex].astype(int).sum() > image[ey, ex].astype(int).sum()
    # a corner well outside the band's feather radius must be untouched
    assert numpy.array_equal(result[50, 50], image[50, 50])


def test_enhancement_never_extends_past_the_image_circle():
    # regression guard: the -2deg horizon allowance lets a near-horizon
    # catalog point stay "visible" even though it projects past the edge
    # of the configured sky circle (fisheye radius grows without bound
    # past the horizon). This index/lat/lon/time is a known case of that:
    # alt is just above -2deg, but its pixel radius exceeds the 500px
    # (1000px diameter) circle while still landing inside the frame. The
    # enhancement must never touch it.
    lat, lon, obstime = -27.0, 153.0, 1767225600.0
    config = _config()
    config['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'] = 1000
    image = numpy.full((1080, 1920, 3), 40, dtype=numpy.uint8)

    near_horizon_point = _GALACTIC_PLANE_CATALOG[84:85]
    alt, az = predictAltAz(near_horizon_point, lat, lon, obstime)
    assert numpy.radians(-2.0) <= alt[0] < 0.0  # confirms this is the case being guarded

    params = (
        config['LENS_AZIMUTH'],
        config['VIRTUALSKY']['LATITUDE_OFFSET'],
        config['VIRTUALSKY']['LONGITUDE_OFFSET'],
        config['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'],
        config['VIRTUALSKY']['OFFSET_X'],
        config['VIRTUALSKY']['OFFSET_Y'],
    )
    x, y = projectToPixels(alt, az, params, 1920, 1080)
    radius = numpy.hypot(x[0] - 960.0, y[0] - 540.0)
    assert radius > 500.0  # confirms it does land outside the 1000px-diameter circle
    assert 0 <= x[0] < 1920 and 0 <= y[0] < 1080  # and still on-frame

    ex, ey = int(round(x[0])), int(round(y[0]))
    result = IndiAllskyMilkyWayStretch(config).apply(image, lat, lon, obstime)

    assert numpy.array_equal(result[ey, ex], image[ey, ex])


def test_enhancement_never_leaks_past_full_resolution_circle_boundary():
    lat, lon, obstime = -27.0, 153.0, 1767225600.0
    config = _config()
    config['IMAGE_STRETCH'].update({
        'MILKYWAY_GAMMA': 4.0,
        'MILKYWAY_BAND_WIDTH': 45.0,
        'MILKYWAY_FEATHER': 500.0,
        'MILKYWAY_SATURATION': 1.0,
    })
    image = numpy.full((1080, 1920, 3), 40, dtype=numpy.uint8)

    result = IndiAllskyMilkyWayStretch(config).apply(image, lat, lon, obstime)
    changed_y, changed_x = numpy.nonzero(numpy.any(result != image, axis=2))
    radius = numpy.hypot(changed_x - 960.0, changed_y - 540.0)

    assert changed_x.size > 0
    assert numpy.all(radius <= 850.0)


def test_milkyway_saturation_boost_increases_color_saturation_in_band():
    lat, lon, obstime = -27.0, 153.0, 1767225600.0
    image = numpy.full((1080, 1920, 3), (30, 30, 60), dtype=numpy.uint8)

    config_no_boost = _config()
    config_no_boost['IMAGE_STRETCH']['MILKYWAY_SATURATION'] = 1.0
    config_boosted = _config()
    config_boosted['IMAGE_STRETCH']['MILKYWAY_SATURATION'] = 2.0

    galactic_center = numpy.array([[266.405, -28.936]])
    alt, az = predictAltAz(galactic_center, lat, lon, obstime)
    x, y = projectToPixels(alt, az, (0.0, 0.0, 0.0, 1700, 0, 0), 1920, 1080)
    ex, ey = int(round(x[0])), int(round(y[0]))

    result_no_boost = IndiAllskyMilkyWayStretch(config_no_boost).apply(image, lat, lon, obstime)
    result_boosted = IndiAllskyMilkyWayStretch(config_boosted).apply(image, lat, lon, obstime)

    sat_no_boost = cv2.cvtColor(result_no_boost, cv2.COLOR_BGR2HSV)[ey, ex, 1]
    sat_boosted = cv2.cvtColor(result_boosted, cv2.COLOR_BGR2HSV)[ey, ex, 1]
    assert int(sat_boosted) > int(sat_no_boost)


def test_milkyway_stretch_never_raises_on_mono_image():
    # the color-only HSV saturation step must be skipped for 2D grayscale
    # frames, not raise
    image = numpy.full((1080, 1920), 40, dtype=numpy.uint8)
    result = IndiAllskyMilkyWayStretch(_config()).apply(image, -27.0, 153.0, 1767225600.0)
    assert numpy.any(result != image)


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


def test_milkyway_gamma_lifts_color_image_luminance_only():
    image = numpy.full((1080, 1920, 3), (20, 30, 50), dtype=numpy.uint8)
    config = _config()
    config['IMAGE_STRETCH']['MILKYWAY_SATURATION'] = 1.0

    result = IndiAllskyMilkyWayStretch(config).apply(image, -27.0, 153.0, 1767225600.0)
    original_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    changed = result_lab[:, :, 0] > original_lab[:, :, 0]

    assert changed.any()
    assert numpy.max(numpy.abs(
        result_lab[:, :, 1].astype(numpy.int16) - original_lab[:, :, 1].astype(numpy.int16))) <= 1
    assert numpy.max(numpy.abs(
        result_lab[:, :, 2].astype(numpy.int16) - original_lab[:, :, 2].astype(numpy.int16))) <= 1


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
    # product requirement: no more than ~300ms per image, even at high
    # resolution. OpenCV pays a one-time per-process init cost on the first
    # call to distanceTransform/resize/LUT/blendLinear; the real process is
    # a long-lived capture daemon that only ever pays that once, so warm up
    # before measuring steady-state cost. Every measured frame must satisfy
    # the product's per-image budget.
    image = numpy.full((2160, 3840, 3), 40, dtype=numpy.uint8)
    enhancer = IndiAllskyMilkyWayStretch(_config())
    for _ in range(2):
        _run_and_time(enhancer, image)  # warm-up, discarded
    timings_ms = [_run_and_time(enhancer, image) for _ in range(7)]
    assert max(timings_ms) < 300.0


def test_milkyway_stretch_never_raises_on_bad_config():
    image = numpy.full((720, 1280, 3), 40, dtype=numpy.uint8)
    config = _config()
    config['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'] = 'not-a-number'
    enhancer = IndiAllskyMilkyWayStretch(config)
    assert enhancer.apply(image, 45.0, -93.0, 1767225600.0) is image


def test_base_stretch_is_disabled_during_moonmode_without_its_own_toggle():
    assert base_stretch_allowed(
        {'MOONMODE': False}, is_night=True, is_moonmode=True) is False


def test_base_stretch_is_enabled_during_moonmode_with_its_own_toggle():
    assert base_stretch_allowed(
        {'MOONMODE': True}, is_night=True, is_moonmode=True) is True


def test_base_stretch_can_run_during_daytime():
    assert base_stretch_allowed(
        {'DAYTIME': True}, is_night=False, is_moonmode=False) is True


def test_base_stretch_is_disabled_when_not_configured():
    assert base_stretch_allowed(
        {}, is_night=True, is_moonmode=False, has_base_stretch=False) is False
