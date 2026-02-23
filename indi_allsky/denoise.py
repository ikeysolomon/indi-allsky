import cv2
import numpy
import logging

from . import constants


logger = logging.getLogger('indi_allsky')


class IndiAllskyDenoise(object):
    """Lightweight image denoising for allsky cameras.

    Provides three Pi-friendly denoising algorithms:
      - median_blur: Fast median filter (removes hot pixels / salt-and-pepper noise)
      - gaussian_blur: Fast Gaussian filter (gentle smoothing)
      - nlm: OpenCV fastNlMeans denoising (best quality, moderate CPU)

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
    # Algorithm: Non-Local Means (best quality, heavier CPU)
    # ------------------------------------------------------------------
    def nlm(self, scidata):
        """Apply OpenCV's fast Non-Local Means denoising.

        The strength parameter controls the filter strength (h):
          - For 8-bit images: h = strength * 3   (range ~3-15)
          - For 16-bit images: h = strength * 300 (scaled to bit depth)
        Strength range: 1-5.

        Works on both grayscale and colour frames.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        is_16bit = scidata.dtype in (numpy.uint16, numpy.int16)
        is_color = len(scidata.shape) == 3

        if is_16bit:
            # fastNlMeans only works on 8-bit; convert, denoise, scale back
            max_val = scidata.max() if scidata.max() > 0 else 1
            scidata_8 = (scidata.astype(numpy.float32) / max_val * 255).astype(numpy.uint8)

            h = strength * 3

            if is_color:
                logger.info('Applying NLM color denoise (16-bit via 8-bit), h=%d', h)
                denoised_8 = cv2.fastNlMeansDenoisingColored(scidata_8, None, h, h, 7, 21)
            else:
                logger.info('Applying NLM denoise (16-bit via 8-bit), h=%d', h)
                denoised_8 = cv2.fastNlMeansDenoising(scidata_8, None, h, 7, 21)

            # Scale back to original range
            denoised = (denoised_8.astype(numpy.float32) / 255.0 * max_val).astype(scidata.dtype)
            return denoised
        else:
            h = strength * 3

            if is_color:
                logger.info('Applying NLM color denoise, h=%d', h)
                return cv2.fastNlMeansDenoisingColored(scidata, None, h, h, 7, 21)
            else:
                logger.info('Applying NLM denoise, h=%d', h)
                return cv2.fastNlMeansDenoising(scidata, None, h, 7, 21)
