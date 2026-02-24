import cv2
import numpy
import logging

from . import constants


logger = logging.getLogger('indi_allsky')


class IndiAllskyDenoise(object):
    """Lightweight image denoising for allsky cameras.

    Provides three Pi-friendly denoising algorithms (quality order):
      - gaussian_blur: Fast Gaussian filter (gentle smoothing)
      - median_blur: Fast median filter (removes hot pixels / salt-and-pepper noise)
      - bilateral: Edge-aware filter (smooths sky while preserving star edges)

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
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR', 20))
            sigma_space = int(self.config.get('BILATERAL_SIGMA_SPACE', 35))
        elif self.night_av[constants.NIGHT_NIGHT]:
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR', 20))
            sigma_space = int(self.config.get('BILATERAL_SIGMA_SPACE', 35))
        else:
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR_DAY', 20))
            sigma_space = int(self.config.get('BILATERAL_SIGMA_SPACE_DAY', 35))

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
