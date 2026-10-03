"""Fast, lens-solution aligned Milky Way enhancement."""

import logging
import time

import cv2
import numpy


logger = logging.getLogger('indi_allsky')


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
        self._image_circle_mask_cache = {}

    def apply(self, image, latitude, longitude, obstime_unix, binning=1, moonmode=False, is_night=True):
        """Apply the enhancement, never raising -- any failure returns
        ``image`` unchanged so a bad frame/config cannot break capture.
        """
        self.last_elapsed_ms = 0.0
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
        gamma = float(settings.get('MILKYWAY_GAMMA', 2.2))
        band_width_deg = float(settings.get('MILKYWAY_BAND_WIDTH', 10.0))
        feather = float(settings.get('MILKYWAY_FEATHER', 60.0))
        if not (numpy.isfinite([gamma, band_width_deg, feather]).all()
                and 1.0 <= gamma <= 4.0 and 1.0 <= band_width_deg <= 45.0
                and 0.0 <= feather <= 500.0):
            raise ValueError('invalid Milky Way enhancement settings')
        if gamma == 1.0:
            return image
        image_height, image_width = image.shape[:2]
        virtualsky = self.config.get('VIRTUALSKY', {})
        output_scale = float(self.config.get('IMAGE_SCALE', 100)) / 100.0
        border = self.config.get('IMAGE_BORDER', {})
        top, right, bottom, left = (int(border.get(key, 0)) for key in ('TOP', 'RIGHT', 'BOTTOM', 'LEFT'))
        output_width, output_height = image_width, image_height
        if output_scale != 1.0:
            output_width = int(image_width * output_scale)
            output_height = int(image_height * output_scale)
            output_width -= output_width % 2
            output_height -= output_height % 2
        if output_width <= 0 or output_height <= 0:
            return image
        scale_x, scale_y = output_width / image_width, output_height / image_height
        output_width += left + right
        output_height += top + bottom
        solved_diameter = float(virtualsky.get('IMAGE_CIRCLE_DIAMETER', 0)) / binning
        if solved_diameter <= 0.0:
            logger.debug('Milky Way enhancement skipped: no image circle diameter configured')
            return image

        params = (
            float(self.config.get('LENS_AZIMUTH', 0.0)),
            float(virtualsky.get('LATITUDE_OFFSET', 0.0)),
            float(virtualsky.get('LONGITUDE_OFFSET', 0.0)),
            solved_diameter,
            float(virtualsky.get('OFFSET_X', 0)) / binning,
            float(virtualsky.get('OFFSET_Y', 0)) / binning,
            float(virtualsky.get('RADIAL_DISTORTION', 0.0)),
        )
        latitude += params[1]
        longitude += params[2]

        from .lens_solver.projection import predictAltAz, precessCatalog, projectToPixels

        catalog = _GALACTIC_PLANE_CATALOG
        if virtualsky.get('PRECESSION', False):
            catalog = precessCatalog(catalog, obstime_unix)
        alt, az = predictAltAz(
            catalog, latitude, longitude, obstime_unix)
        lens_altitude = float(self.config.get('LENS_ALTITUDE', 90.0))
        pointing_azimuth = float(virtualsky.get('POINTING_AZIMUTH', 0.0))
        x, y = projectToPixels(
            alt, az, params, output_width, output_height,
            lens_altitude=lens_altitude, pointing_azimuth=pointing_azimuth)
        model = virtualsky.get('CALIBRATION')
        if virtualsky.get('CALIBRATION_ENABLED', False) and model is not None:
            from .lens_solver.calibration import displacement, pipelineSignature, validateCalibration

            geometry = [params[0], params[1], params[2], solved_diameter * binning,
                        params[4] * binning, params[5] * binning,
                        lens_altitude, pointing_azimuth, params[6],
                        int(bool(virtualsky.get('PRECESSION', False)))]
            if not validateCalibration(model):
                raise ValueError('invalid lens calibration')
            saved_geometry = model['geometry'] + ([0, 0] if model['version'] == 1 else [])
            if saved_geometry != geometry:
                raise ValueError('lens calibration geometry has changed')
            if (model['pipeline'] != pipelineSignature(self.config)
                    or model['image_size'] != [output_width, output_height]):
                raise ValueError('lens calibration image pipeline has changed')
            center = numpy.array([output_width / 2.0 + params[4], output_height / 2.0 - params[5]])
            points = numpy.column_stack((x, y))
            points += displacement((points - center) / (solved_diameter / 2.0), model) * (solved_diameter / 2.0)
            x, y = points.T
        x, y = (x - left) / scale_x, (y - top) / scale_y
        center_x = (output_width / 2.0 + params[4] - left) / scale_x
        center_y = (output_height / 2.0 - params[5] - top) / scale_y
        radius_x, radius_y = solved_diameter / (2.0 * scale_x), solved_diameter / (2.0 * scale_y)
        diameter = 2.0 * max(radius_x, radius_y)

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

        feather_px = feather * scale
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
        circle_cx = center_x * scale
        circle_cy = center_y * scale
        yy, xx = numpy.ogrid[:mask_height, :mask_width]
        outside_circle = ((xx - circle_cx) / (radius_x * scale)) ** 2 + ((yy - circle_cy) / (radius_y * scale)) ** 2 > 1.0
        mask[outside_circle] = 0

        if scale < 1.0:
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

        # Linear upscaling can interpolate non-zero alpha just outside the
        # low-resolution circle edge. Reapply the exact full-resolution
        # boundary so no enhancement reaches invalid camera pixels.
        circle_key = (image_width, image_height, center_x, center_y, radius_x, radius_y)
        circle_mask = self._image_circle_mask_cache.get(circle_key)
        if circle_mask is None:
            yy, xx = numpy.ogrid[:image_height, :image_width]
            circle_mask = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0).astype(numpy.uint8) * 255
            self._image_circle_mask_cache.clear()
            self._image_circle_mask_cache[circle_key] = circle_mask
        mask = cv2.bitwise_and(mask, circle_mask)

        if not numpy.any(mask):
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
