import cv2
import numpy
import logging

from . import constants


logger = logging.getLogger('indi_allsky')


class IndiAllskyDenoise(object):
    """Lightweight image denoising for allsky cameras.

    Provides four denoising algorithms (quality order):
      - gaussian_blur: Fast Gaussian filter (gentle smoothing)
      - median_blur: Fast median filter (removes hot pixels / salt-and-pepper noise)
      - bilateral: Edge-aware filter (smooths sky while preserving star edges)
      - nlm_luma: Non-Local Means, luminance-only (best quality, moderate cost)

    Each algorithm respects a configurable strength parameter so users
    can tune the trade-off between noise reduction and star preservation.

    Temporal averaging is handled separately by the stacking system.
    """

    def __init__(self, config, night_av):
        self.config = config
        self.night_av = night_av


    def _get_strength(self):
        """Return the effective denoise strength (int) respecting night/day config."""
        if self.config.get('USE_NIGHT_COLOR', True):
            return int(self.config.get('IMAGE_DENOISE_STRENGTH', 3))

        if self.night_av[constants.NIGHT_NIGHT]:
            return int(self.config.get('IMAGE_DENOISE_STRENGTH', 3))

        # daytime
        return int(self.config.get('IMAGE_DENOISE_STRENGTH_DAY', 3))


    def _get_bilateral_sigma(self):
        """Return (sigmaColor, sigmaSpace) for the bilateral filter."""
        if self.config.get('USE_NIGHT_COLOR', True):
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR', 10))
            sigma_space = int(self.config.get('BILATERAL_SIGMA_SPACE', 15))
        elif self.night_av[constants.NIGHT_NIGHT]:
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR', 10))
            sigma_space = int(self.config.get('BILATERAL_SIGMA_SPACE', 15))
        else:
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR_DAY', 10))
            sigma_space = int(self.config.get('BILATERAL_SIGMA_SPACE_DAY', 15))

        return max(1, sigma_color), max(1, sigma_space)


    # ------------------------------------------------------------------
    # Algorithm: Median Blur
    # ------------------------------------------------------------------
    def median_blur(self, scidata):
        """Apply a fast median blur.

        The strength parameter maps to the kernel size:
          1 → 3x3,  2 → 5x5,  3 → 7x7  (formula: ksize = strength * 2 + 1)
        Strength range: 1-5.  Higher values smear stars.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        # Clamp strength to sane range
        strength = max(1, min(strength, 5))

        ksize = strength * 2 + 1  # always odd: 3, 5, 7, 9, 11

        logger.info('Applying median blur denoise, ksize=%d', ksize)

        return cv2.medianBlur(scidata, ksize)


    # ------------------------------------------------------------------
    # Algorithm: Gaussian Blur
    # ------------------------------------------------------------------
    def gaussian_blur(self, scidata):
        """Apply a fast Gaussian blur.

        The strength parameter maps to the kernel size:
          1 → 3x3,  2 → 5x5,  3 → 7x7
        Sigma is computed automatically by OpenCV (sigmaX=0).
        Strength range: 1-5.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        ksize = strength * 2 + 1

        logger.info('Applying gaussian blur denoise, ksize=%d', ksize)

        return cv2.GaussianBlur(scidata, (ksize, ksize), 0)


    # ------------------------------------------------------------------
    # Algorithm: Bilateral Filter (edge-aware, best quality)
    # ------------------------------------------------------------------
    def bilateral(self, scidata):
        """Apply an edge-aware bilateral filter.

        Smooths areas of similar brightness (noisy sky background) while
        preserving sharp intensity transitions (star edges).  Much faster
        than Non-Local Means but higher quality than Gaussian/Median for
        astro images.

        The strength parameter controls the filter size (d):
          d = strength * 2 + 1   (diameter: 3, 5, 7, 9, 11)
        sigmaColor and sigmaSpace are user-configurable:
          sigmaColor controls how much difference in brightness is tolerated
          sigmaSpace controls how far away pixels can influence
        Lower sigmaColor preserves more edges.  Strength range: 1-5.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        d = strength * 2 + 1
        sigma_color, sigma_space = self._get_bilateral_sigma()

        needs_conversion = scidata.dtype not in (numpy.uint8, numpy.float32)

        if needs_conversion:
            # bilateralFilter supports uint8 and float32.
            # OpenCV float32 bilateral is optimized for 0.0-1.0 range.
            # Normalize data to 0-1, scale sigmaColor to match (divide
            # by 255 since user-facing sigma is calibrated for 0-255).
            # sigmaSpace is in pixel units and needs no adjustment.
            if numpy.issubdtype(scidata.dtype, numpy.integer):
                dtype_max = numpy.float32(numpy.iinfo(scidata.dtype).max)
            else:
                dtype_max = numpy.float32(1.0)

            sigma_color_norm = float(sigma_color) / 255.0
            scidata_f32 = scidata.astype(numpy.float32) / dtype_max

            logger.info('Applying bilateral denoise (float32 0-1), d=%d sigmaColor=%.4f sigmaSpace=%d', d, sigma_color_norm, sigma_space)
            denoised_f32 = cv2.bilateralFilter(scidata_f32, d, sigma_color_norm, float(sigma_space))

            return numpy.clip(numpy.rint(denoised_f32 * dtype_max), 0, float(dtype_max)).astype(scidata.dtype)
        else:
            logger.info('Applying bilateral denoise, d=%d sigmaColor=%d sigmaSpace=%d', d, sigma_color, sigma_space)
            return cv2.bilateralFilter(scidata, d, sigma_color, sigma_space)


    # ------------------------------------------------------------------
    # Algorithm: Non-Local Means — luminance-only, tuned (best quality)
    # ------------------------------------------------------------------
    def nlm_luma(self, scidata):
        """Apply Non-Local Means denoising (luminance-only, tight search).

        Uses OpenCV's C-optimised NLM engine but applies two key speedups
        over the naive ``fastNlMeansDenoisingColored`` call:

          1. Colour images: only denoise the luminance (Y) channel.
             Chrominance carries little perceptible noise so this gives
             a ~3× speed-up with no visible quality loss.
          2. Tighter search window: ``searchWindowSize = strength * 4 + 3``
             (11 at strength 2, 15 at strength 3).  Much faster than the
             default 21 and still covers the useful self-similarity range
             for allsky images.

        The strength parameter controls:
          h (filter strength)    = strength * 3 + 1   (4, 7, 10, 13, 16)
          templateWindowSize     = strength * 2 + 1   (3, 5, 7, 9, 11)
          searchWindowSize       = strength * 4 + 3   (7, 11, 15, 19, 23)

        Typical Pi 4 performance at strength 3: ~1-2 s.
        Strength range: 1-5.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        h = strength * 3 + 1
        template_ws = strength * 2 + 1  # must be odd
        search_ws = strength * 4 + 3    # must be odd

        # --- Handle bit depth ------------------------------------------
        original_dtype = scidata.dtype
        needs_conversion = original_dtype not in (numpy.uint8,)

        if needs_conversion:
            # fastNlMeansDenoising only accepts uint8; normalise to 0-255
            if numpy.issubdtype(original_dtype, numpy.integer):
                dtype_max = numpy.float32(numpy.iinfo(original_dtype).max)
            else:
                dtype_max = numpy.float32(1.0)
            scidata_8 = numpy.clip(
                scidata.astype(numpy.float32) / dtype_max * 255.0,
                0, 255,
            ).astype(numpy.uint8)
        else:
            dtype_max = numpy.float32(255.0)
            scidata_8 = scidata

        # --- Colour: luminance-only denoise (~3× faster) ---------------
        if scidata_8.ndim == 3:
            ycrcb = cv2.cvtColor(scidata_8, cv2.COLOR_BGR2YCrCb)

            logger.info(
                'Applying NLM denoise (luma-only), h=%d template=%d search=%d',
                h, template_ws, search_ws,
            )
            ycrcb[:, :, 0] = cv2.fastNlMeansDenoising(
                ycrcb[:, :, 0], None, h, template_ws, search_ws,
            )
            denoised_8 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            logger.info(
                'Applying NLM denoise (mono), h=%d template=%d search=%d',
                h, template_ws, search_ws,
            )
            denoised_8 = cv2.fastNlMeansDenoising(
                scidata_8, None, h, template_ws, search_ws,
            )

        # --- Convert back to original dtype ----------------------------
        if needs_conversion:
            return numpy.clip(
                numpy.rint(denoised_8.astype(numpy.float32) / 255.0 * dtype_max),
                0, float(dtype_max),
            ).astype(original_dtype)
        else:
            return denoised_8
