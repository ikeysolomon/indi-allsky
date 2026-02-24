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
    # Algorithm: Median Blur (conditional — hot pixel removal)
    # ------------------------------------------------------------------
    def median_blur(self, scidata):
        """Apply a conditional median filter (hot/cold pixel removal).

        Computes a local median for each pixel.  Only pixels that differ
        from their neighbourhood median by more than a threshold are
        replaced; clean pixels pass through untouched.  This is the same
        approach used by G'MIC's "Remove Hot Pixels" filter.

        The strength parameter controls the kernel size:
          1 → 3×3,  2 → 5×5,  3 → 7×7  (ksize = strength * 2 + 1)

        The threshold is derived from the image dtype:
          uint8  → 15 counts
          uint16 → 15 × 257 ≈ 3855 counts   (same proportion of range)
          float  → 15 / 255 ≈ 0.059

        Strength range: 1-5.  Higher values use a larger neighbourhood.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        # Clamp strength to sane range
        strength = max(1, min(strength, 5))

        ksize = strength * 2 + 1  # always odd: 3, 5, 7, 9, 11

        # Compute local median
        median = cv2.medianBlur(scidata, ksize)

        # Threshold: ~6 % of the dynamic range
        if scidata.dtype == numpy.uint8:
            threshold = numpy.uint8(15)
        elif scidata.dtype == numpy.uint16:
            threshold = numpy.uint16(3855)
        elif scidata.dtype == numpy.int16:
            threshold = numpy.int16(1928)
        else:
            # float
            threshold = numpy.float32(15.0 / 255.0)

        # Only replace outlier pixels; leave clean pixels untouched
        diff = cv2.absdiff(scidata, median)
        outlier_mask = diff > threshold
        result = scidata.copy()
        result[outlier_mask] = median[outlier_mask]

        n_replaced = int(numpy.count_nonzero(outlier_mask))
        logger.info('Applying conditional median denoise, ksize=%d threshold=%s replaced=%d pixels',
                     ksize, str(threshold), n_replaced)

        return result


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
