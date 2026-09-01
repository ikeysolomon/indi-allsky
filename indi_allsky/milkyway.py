"""Fast, lens-solution aligned Milky Way enhancement for stretched images."""

import logging
import time

import cv2
import numpy

from .lens_solver.projection import predictAltAz
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

        alt, az = predictAltAz(
            _GALACTIC_PLANE_CATALOG, latitude, longitude, obstime_unix)
        x, y = projectToPixels(alt, az, params, image_width, image_height)

        # Rasterize at a bounded resolution.  This makes the mask generation
        # cost predictable on high-resolution camera frames.
        scale = min(1.0, 1024.0 / max(image_width, image_height))
        mask_width = max(1, int(round(image_width * scale)))
        mask_height = max(1, int(round(image_height * scale)))
        mask = numpy.zeros((mask_height, mask_width), dtype=numpy.uint8)

        # Only a thin centerline is rasterized -- a filled band would leave
        # every interior pixel at distance 0 from the (bitwise-inverted)
        # mask, producing a hard, flat-alpha plateau across its full width
        # no matter how much feather is applied. Measuring distance from
        # the centerline instead lets alpha taper continuously across the
        # entire band, so there is no hard edge anywhere.
        band_width_deg = float(settings.get('MILKYWAY_BAND_WIDTH', 10.0))
        half_width_px = max(1.0, diameter * band_width_deg * numpy.pi / 360.0 * scale / 2.0)
        centerline_px = max(1, int(round(scale)))
        points = numpy.rint(numpy.column_stack((x * scale, y * scale))).astype(numpy.int32)
        visible = alt >= numpy.radians(-2.0)
        max_segment_length = diameter * scale * 0.12
        segment = []
        for index, point in enumerate(points):
            if (not visible[index] or
                    (segment and numpy.hypot(*(point - segment[-1])) > max_segment_length)):
                if len(segment) > 1:
                    # LINE_8, not LINE_AA: distanceTransform needs an exact
                    # 255/0 mask to find its zero reference points; the
                    # smoothstep falloff below is what actually smooths it.
                    cv2.polylines(mask, [numpy.asarray(segment)], False, 255, centerline_px, cv2.LINE_8)
                segment = []
            if visible[index]:
                segment.append(point)
        if len(segment) > 1:
            cv2.polylines(mask, [numpy.asarray(segment)], False, 255, centerline_px, cv2.LINE_8)

        feather_px = float(settings.get('MILKYWAY_FEATHER', 60.0)) * scale
        falloff_px = half_width_px + feather_px
        if falloff_px > 0.0:
            # Smoothstep of distance-from-centerline is a cheap, seamless
            # stand-in for a large Gaussian blur.
            distance = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
            t = numpy.clip(1.0 - distance / falloff_px, 0.0, 1.0)
            mask = (t * t * (3.0 - 2.0 * t) * 255.0).astype(numpy.uint8)

        # the enhancement must never touch pixels outside the camera's own
        # valid sky circle -- the -2deg horizon allowance (and the feather
        # falloff itself) can otherwise push it past the circle edge, and
        # this must not depend on some other pipeline stage (e.g. circular
        # cropping) to clean it up
        circle_cx = (image_width / 2.0 + params[4]) * scale
        circle_cy = (image_height / 2.0 - params[5]) * scale
        circle_radius = (diameter / 2.0) * scale
        yy, xx = numpy.ogrid[:mask_height, :mask_width]
        outside_circle = (xx - circle_cx) ** 2 + (yy - circle_cy) ** 2 > circle_radius ** 2
        mask[outside_circle] = 0

        if scale < 1.0:
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

        gamma = float(settings.get('MILKYWAY_GAMMA', 2.2))
        if gamma <= 1.0 or not numpy.any(mask):
            logger.debug('Milky Way enhancement skipped: band not visible or gamma is a no-op')
            return image

        alpha = mask.astype(numpy.float32) / 255.0
        if image.dtype == numpy.uint8:
            lut = numpy.clip(
                numpy.power(numpy.arange(256, dtype=numpy.float32) / 255.0, 1.0 / gamma) * 255.0,
                0.0, 255.0).astype(numpy.uint8)

            # deliberately full resolution: this is a quality-focused
            # enhancement (dust-lane/color detail), not the geometry mask,
            # so it must not be softened by a downscale/upscale round-trip
            enhanced = cv2.LUT(image, lut)

            if image.ndim == 3:
                # local contrast on luminance brings out dust-lane definition
                # that a flat gamma lift alone washes out (gamma brightens
                # the dust lanes almost as much as their surroundings)
                clip_limit = float(self.config.get('CLAHE_CLIPLIMIT', 3.0))
                grid_size = int(self.config.get('CLAHE_GRIDSIZE', 8))
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # saturation boost brings out the reds/pinks of emission
                # nebulosity along the plane, which gamma alone (the same
                # curve applied identically to every channel) does not
                saturation = float(settings.get('MILKYWAY_SATURATION', 1.4))
                if saturation != 1.0:
                    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
                    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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
