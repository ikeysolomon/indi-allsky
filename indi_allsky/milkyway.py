"""Fast, lens-solution aligned Milky Way enhancement for stretched images."""

import logging
import time

import cv2
import numpy

from .lens_solver.projection import projectToPixels


logger = logging.getLogger('indi_allsky')


def stretch_eligibility(stretch_config, is_night, is_moonmode, has_base_stretch=True, lens_solved=True):
    """Decide, independently, whether the base histogram stretch and the
    Milky Way enhancement should run for the current frame. Enabling the
    Milky Way Moon Mode toggle must never force the (unrelated) base
    stretch to run during Moon Mode, and vice versa; a base stretch that
    isn't even configured (``has_base_stretch=False``) must not block the
    Milky Way enhancement either. Without a solved lens (``lens_solved=False``)
    the projection geometry is untrustworthy, so the band would land in the
    wrong place -- it must never run in that case.
    """
    if is_night:
        base_allowed = has_base_stretch and (not is_moonmode or bool(stretch_config.get('MOONMODE')))
    else:
        base_allowed = has_base_stretch and bool(stretch_config.get('DAYTIME'))

    milkyway_enabled = bool(stretch_config.get('MILKYWAY_ENABLE', False))
    milkyway_allowed = (
        is_night
        and milkyway_enabled
        and lens_solved
        and (not is_moonmode or bool(stretch_config.get('MILKYWAY_MOONMODE', False)))
    )

    return base_allowed, milkyway_allowed


# IAU 1958 equatorial(J2000)-to-galactic rotation matrix (standard "A_G").
# The Galactic plane is sampled once per degree; that is dense enough for a
# smooth rasterized band. Since A_G is orthogonal, applying it (without
# transposing) to a row vector via `@` yields the galactic-to-equatorial
# transform: e_row = g_row @ A_G  <=>  e_col = A_G.T @ g_col = A_G^-1 @ g_col.
_EQUATORIAL_TO_GALACTIC = numpy.array((
    (-0.0548755604, -0.8734370902, -0.4838350155),
    (0.4941094279, -0.4448296300, 0.7469822445),
    (-0.8676661490, -0.1980763734, 0.4559837762),
), dtype=numpy.float64)
_GALACTIC_LONGITUDES = numpy.radians(numpy.arange(-180.0, 181.0, 1.0))


def _galactic_plane_catalog():
    galactic_vectors = numpy.column_stack((
        numpy.cos(_GALACTIC_LONGITUDES),
        numpy.sin(_GALACTIC_LONGITUDES),
        numpy.zeros_like(_GALACTIC_LONGITUDES),
    ))
    equatorial_vectors = galactic_vectors @ _EQUATORIAL_TO_GALACTIC
    ra = numpy.degrees(numpy.arctan2(
        equatorial_vectors[:, 1], equatorial_vectors[:, 0])) % 360.0
    dec = numpy.degrees(numpy.arcsin(numpy.clip(equatorial_vectors[:, 2], -1.0, 1.0)))
    return numpy.column_stack((ra, dec))


_GALACTIC_PLANE_CATALOG = _galactic_plane_catalog()


def _predict_alt_az(catalog, latitude, longitude, obstime_unix):
    """Fast equivalent of the VirtualSky alt/az transform for rendering."""
    days_since_j2000 = (obstime_unix - 946728000.0) / 86400.0
    gmst = numpy.radians((280.46061837 + 360.98564736629 * days_since_j2000) % 360.0)
    ra = numpy.radians(catalog[:, 0])
    dec = numpy.radians(catalog[:, 1])
    latitude_rad = numpy.radians(latitude)
    hour_angle = gmst + numpy.radians(longitude) - ra
    alt = numpy.arcsin(numpy.clip(
        numpy.sin(dec) * numpy.sin(latitude_rad) +
        numpy.cos(dec) * numpy.cos(latitude_rad) * numpy.cos(hour_angle),
        -1.0, 1.0))
    az = numpy.arctan2(
        -numpy.cos(dec) * numpy.sin(hour_angle),
        numpy.sin(dec) * numpy.cos(latitude_rad) -
        numpy.cos(dec) * numpy.sin(latitude_rad) * numpy.cos(hour_angle))
    return alt, numpy.mod(az, 2.0 * numpy.pi)


class IndiAllskyMilkyWayStretch(object):
    """Create and apply a feathered Galactic-plane enhancement mask."""

    def __init__(self, config):
        self.config = config
        self.last_elapsed_ms = 0.0

    def apply(self, image, latitude, longitude, obstime_unix, binning=1, moonmode=False, is_night=True):
        """Apply the enhancement, never raising -- any failure returns
        ``image`` unchanged so a bad frame/config cannot break capture.
        """
        settings = self.config.get('IMAGE_STRETCH', {})
        if not settings.get('MILKYWAY_ENABLE', False):
            return image

        # an unsolved lens has no trustworthy azimuth/offset geometry -- the
        # band would render, just in the wrong place -- so this is the one
        # guard that may never be bypassed or defaulted True
        if not self.config.get('LENS_SOLVED', False):
            logger.debug('Milky Way enhancement skipped: lens has not been plate solved')
            return image

        # the Milky Way is never visible in daylight; this must be checked
        # independently of the base stretch's own daytime toggle
        if not is_night:
            return image

        # moonlight washes out the Milky Way; skip unless the user opted
        # in via its own toggle, independent of the base stretch's Moon
        # Mode setting.
        if moonmode and not settings.get('MILKYWAY_MOONMODE', False):
            logger.debug('Milky Way enhancement skipped: moon mode active')
            return image

        try:
            return self._apply(image, settings, latitude, longitude, obstime_unix, binning)
        except Exception as e:
            logger.warning('Milky Way enhancement skipped: %s', str(e))
            return image

    def _apply(self, image, settings, latitude, longitude, obstime_unix, binning):
        t_start = time.monotonic()
        image_height, image_width = image.shape[:2]
        virtualsky = self.config.get('VIRTUALSKY', {})
        diameter = float(virtualsky.get('IMAGE_CIRCLE_DIAMETER', 0)) / binning
        if diameter <= 0.0:
            logger.debug('Milky Way enhancement skipped: no image circle diameter configured')
            return image

        params = (
            float(self.config.get('LENS_AZIMUTH', 0.0)),
            float(virtualsky.get('LATITUDE_OFFSET', 0.0)),
            float(virtualsky.get('LONGITUDE_OFFSET', 0.0)),
            diameter,
            float(virtualsky.get('OFFSET_X', 0)) / binning,
            float(virtualsky.get('OFFSET_Y', 0)) / binning,
        )
        latitude += params[1]
        longitude += params[2]

        alt, az = _predict_alt_az(
            _GALACTIC_PLANE_CATALOG, latitude, longitude, obstime_unix)
        x, y = projectToPixels(alt, az, params, image_width, image_height)

        # Rasterize at a bounded resolution.  This makes the mask generation
        # cost predictable on high-resolution camera frames.
        scale = min(1.0, 1024.0 / max(image_width, image_height))
        mask_width = max(1, int(round(image_width * scale)))
        mask_height = max(1, int(round(image_height * scale)))
        mask = numpy.zeros((mask_height, mask_width), dtype=numpy.uint8)

        band_width_deg = float(settings.get('MILKYWAY_BAND_WIDTH', 14.0))
        band_width_px = max(1, int(round(diameter * band_width_deg * numpy.pi / 360.0 * scale)))
        points = numpy.rint(numpy.column_stack((x * scale, y * scale))).astype(numpy.int32)
        visible = alt >= numpy.radians(-2.0)
        max_segment_length = diameter * scale * 0.12
        segment = []
        for index, point in enumerate(points):
            if (not visible[index] or
                    (segment and numpy.hypot(*(point - segment[-1])) > max_segment_length)):
                if len(segment) > 1:
                    cv2.polylines(mask, [numpy.asarray(segment)], False, 255, band_width_px, cv2.LINE_AA)
                segment = []
            if visible[index]:
                segment.append(point)
        if len(segment) > 1:
            cv2.polylines(mask, [numpy.asarray(segment)], False, 255, band_width_px, cv2.LINE_AA)

        feather_px = float(settings.get('MILKYWAY_FEATHER', 80.0)) * scale
        if feather_px > 0.0:
            # A distance-based smoothstep ramp keeps the band opaque while
            # gently, smoothly fading its edge (no visible seam where the
            # solid band meets the fade, unlike a plain linear ramp), and is
            # substantially cheaper than a large Gaussian blur.
            distance = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
            t = numpy.clip(1.0 - distance / feather_px, 0.0, 1.0)
            mask = (t * t * (3.0 - 2.0 * t) * 255.0).astype(numpy.uint8)
        if scale < 1.0:
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

        gamma = float(settings.get('MILKYWAY_GAMMA', 1.35))
        if gamma <= 1.0 or not numpy.any(mask):
            logger.debug('Milky Way enhancement skipped: band not visible or gamma is a no-op')
            return image

        alpha = mask.astype(numpy.float32) / 255.0
        if image.dtype == numpy.uint8:
            lut = numpy.clip(
                numpy.power(numpy.arange(256, dtype=numpy.float32) / 255.0, 1.0 / gamma) * 255.0,
                0.0, 255.0).astype(numpy.uint8)
            enhanced = cv2.LUT(image, lut)
            result = cv2.blendLinear(image, enhanced, 1.0 - alpha, alpha)
            self.last_elapsed_ms = (time.monotonic() - t_start) * 1000.0
            return result

        dtype_max = numpy.iinfo(image.dtype).max if numpy.issubdtype(image.dtype, numpy.integer) else 1.0
        normalized = image.astype(numpy.float32) / dtype_max
        enhanced = numpy.power(normalized, 1.0 / gamma) * dtype_max
        if image.ndim == 3:
            alpha = alpha[:, :, numpy.newaxis]
        result = image.astype(numpy.float32) * (1.0 - alpha) + enhanced * alpha
        self.last_elapsed_ms = (time.monotonic() - t_start) * 1000.0
        return numpy.clip(result, 0.0, dtype_max).astype(image.dtype)
