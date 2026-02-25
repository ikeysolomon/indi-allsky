import cv2
import numpy
import logging

from . import constants


logger = logging.getLogger('indi_allsky')


class IndiAllskyDenoise(object):
    """Lightweight image denoising for allsky cameras.

    Provides four denoising algorithms (quality order):
      - gaussian_blur: Luminance-masked Gaussian filter (smooths sky, preserves bright stars)
      - median_blur: fixed-threshold median filter (removes salt-and-pepper noise)
      - bilateral: Edge-aware filter (smooths sky while preserving star edges)
      - wavelet: BayesShrink wavelet denoise (frequency-domain, best quality, requires PyWavelets)

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


    def _medianBlur(self, img, ksize):
        """cv2.medianBlur only supports CV_8U for multi-channel images in newer OpenCV.
        Split into per-channel blurs when the image is 16-bit (or deeper) multi-channel."""
        def _blur_channel(ch):
            # Fast path for 8-bit single-channel
            if ch.dtype == numpy.uint8:
                return cv2.medianBlur(ch, ksize)

            # Integer types (commonly uint16) — scale down to 8-bit, blur, scale back
            if numpy.issubdtype(ch.dtype, numpy.integer):
                # Use a divisor that maps 0..65535 -> 0..255 approximately
                divisor = 257
                ch8 = (ch.astype(numpy.uint32) // divisor).astype(numpy.uint8)
                b8 = cv2.medianBlur(ch8, ksize)
                return (b8.astype(numpy.uint32) * divisor).astype(ch.dtype)

            # Floating types: scale to 0..255, blur, then rescale back
            ch8 = numpy.clip((ch * 255.0), 0, 255).astype(numpy.uint8)
            b8 = cv2.medianBlur(ch8, ksize)
            return (b8.astype(numpy.float32) / 255.0).astype(ch.dtype)

        # Single-channel image
        if img.ndim == 2:
            return _blur_channel(img)

        # Multi-channel: process each channel independently and merge
        channels = cv2.split(img)
        blurred = [_blur_channel(ch) for ch in channels]
        return cv2.merge(blurred)

    # ------------------------------------------------------------------
    # Algorithm: Median Blur (fixed-threshold — salt-and-pepper removal)
    # ------------------------------------------------------------------
    def median_blur(self, scidata):
        """Apply a fixed-threshold median filter to remove salt-and-pepper noise.

        Computes the local median for each pixel's neighbourhood, then
        replaces only pixels that deviate from the median by more than a
        fixed ADU threshold.  Multi-pixel features (stars, gradients) are
        preserved because their local median is representative of the
        feature itself, not of the sky background.

        Strength controls kernel size:
          1 → 3×3,  2 → 5×5,  3 → 7×7,  4 → 9×9,  5 → 11×11
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))
        ksize = strength * 2 + 1  # always odd: 3, 5, 7, 9, 11

        if scidata.dtype == numpy.uint8:
            threshold = numpy.float32(5.0)
        elif scidata.dtype in (numpy.uint16, numpy.int16):
            threshold = numpy.float32(5.0 * 257.0)  # ~1285 in 16-bit
        else:
            threshold = numpy.float32(5.0 / 255.0)

        local_median = self._medianBlur(scidata, ksize)

        diff = cv2.absdiff(scidata, local_median).astype(numpy.float32)
        outlier_mask = diff > threshold

        result = scidata.copy()
        result[outlier_mask] = local_median[outlier_mask]

        n_replaced = int(numpy.count_nonzero(outlier_mask))
        logger.info('Applying median denoise, ksize=%d threshold=%.1f replaced=%d pixels',
                    ksize, float(threshold), n_replaced)

        return result


    # ------------------------------------------------------------------
    # Algorithm: Gaussian Blur (luminance-masked — preserves bright features)
    # ------------------------------------------------------------------
    def gaussian_blur(self, scidata):
        """Apply a luminance-masked Gaussian blur.

        Smooths dark sky regions while preserving bright features (stars,
        planets, moon).  Works by:

          1. Compute a Gaussian-blurred copy of the image.
          2. Determine a brightness threshold at the 85th percentile.
          3. Build a soft blend mask: pixels well above the threshold
             keep their original values; pixels well below get the
             blurred values; a smooth transition (~10% of dtype range)
             avoids hard edges.

        The strength parameter maps to Gaussian sigma (linear: step=2.4, max=12):
          1 → σ=2.4,  2 → σ=4.8,  3 → σ=7.2,  4 → σ=9.6,  5 → σ=12.0
        Kernel size is derived automatically as 6σ (rounded to odd).
        Strength range: 1-5.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        sigma = strength * 2.4

        blurred = cv2.GaussianBlur(scidata, (0, 0), sigma)

        # Determine dtype-aware threshold at 85th percentile
        if scidata.dtype == numpy.uint8:
            dtype_max = numpy.float32(255.0)
        elif scidata.dtype in (numpy.uint16, numpy.int16):
            dtype_max = numpy.float32(65535.0)
        else:
            dtype_max = numpy.float32(1.0)

        threshold = numpy.float32(numpy.percentile(scidata, 85))

        # Soft transition width: 10% of dtype range (avoids hard edges)
        transition = numpy.float32(dtype_max * 0.10)
        transition = max(transition, numpy.float32(1.0))  # floor for uint8

        # Build soft blend mask: 0.0 = use blurred, 1.0 = keep original
        scidata_f32 = scidata.astype(numpy.float32)
        # alpha ramps from 0→1 over the transition band above threshold
        alpha = numpy.clip((scidata_f32 - threshold) / transition, 0.0, 1.0)

        # Blend: result = alpha * original + (1 - alpha) * blurred
        result_f32 = alpha * scidata_f32 + (numpy.float32(1.0) - alpha) * blurred.astype(numpy.float32)

        result = numpy.clip(result_f32, 0, float(dtype_max)).astype(scidata.dtype)

        n_preserved = int(numpy.count_nonzero(alpha > 0.5))
        logger.info('Applying luminance-masked gaussian denoise, sigma=%.1f threshold=%.0f preserved=%d bright pixels',
                     sigma, float(threshold), n_preserved)

        return result


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
    # Algorithm: Wavelet Denoise (BayesShrink — frequency-domain, best quality)
    # ------------------------------------------------------------------
    def wavelet(self, scidata):
        """Apply wavelet-based denoising with BayesShrink adaptive thresholding.

        Decomposes the image into frequency bands using the Discrete
        Wavelet Transform (DWT), estimates noise in the finest detail
        coefficients, and shrinks noisy coefficients using BayesShrink
        adaptive soft thresholding.  This naturally separates:

          - Stars (fine-scale signal — preserved)
          - Sky gradients (coarse-scale — preserved)
          - Noise (medium-scale random — suppressed)

        The strength parameter scales the BayesShrink threshold linearly:
          1 → 1.2×,  2 → 2.4×,  3 → 3.6×,  4 → 4.8×,  5 → 6.0×

        Wavelet: Daubechies-4 (db4), levels: auto (3-4), soft thresholding.
        Requires PyWavelets (pywt).  Strength range: 1-5.
        """
        try:
            import pywt
        except ImportError:
            logger.warning('PyWavelets (pywt) is not installed — falling back to median denoise')
            # Fall back to the fast median-based denoise to provide a usable
            # result when the optional dependency is missing.
            return self.median_blur(scidata)

        strength = self._get_strength()

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        # Scale factor applied to BayesShrink threshold (linear: step=1.2, max=6.0):
        #   1 → 1.2,  2 → 2.4,  3 → 3.6,  4 → 4.8,  5 → 6.0
        scale = strength * 1.2

        # Determine dtype range for normalization
        if numpy.issubdtype(scidata.dtype, numpy.integer):
            dtype_max = float(numpy.iinfo(scidata.dtype).max)
        else:
            dtype_max = 1.0

        orig_dtype = scidata.dtype

        # Auto decomposition levels: 3-4 based on smallest image dimension
        min_dim = min(scidata.shape[0], scidata.shape[1])
        max_level = pywt.dwt_max_level(min_dim, pywt.Wavelet('db4').dec_len)
        levels = min(max(max_level, 1), 4)

        def _denoise_channel(channel):
            """Denoise a single 2D channel using BayesShrink."""
            # Normalize to 0-1 float64 for wavelet precision
            data = channel.astype(numpy.float64) / dtype_max

            # Forward DWT
            coeffs = pywt.wavedec2(data, 'db4', level=levels)

            # Estimate noise sigma from finest detail coefficients (HH band)
            # MAD estimator: sigma = median(|d|) / 0.6745
            detail_hh = coeffs[-1][2]  # HH = diagonal detail at finest level
            sigma_noise = numpy.median(numpy.abs(detail_hh)) / 0.6745

            if sigma_noise < 1e-10:
                # No measurable noise
                return channel

            # Apply BayesShrink to each detail level
            denoised_coeffs = [coeffs[0]]  # keep approximation coefficients untouched
            for i in range(1, len(coeffs)):
                new_details = []
                for detail_band in coeffs[i]:
                    # BayesShrink threshold: sigma_noise^2 / sigma_signal
                    sigma_band = numpy.sqrt(max(numpy.var(detail_band) - sigma_noise ** 2, 0))
                    if sigma_band < 1e-10:
                        threshold = numpy.max(numpy.abs(detail_band))  # shrink everything
                    else:
                        threshold = (sigma_noise ** 2) / sigma_band

                    # Scale by user strength
                    threshold *= scale

                    # Soft thresholding
                    new_details.append(pywt.threshold(detail_band, threshold, mode='soft'))

                denoised_coeffs.append(tuple(new_details))

            # Inverse DWT
            reconstructed = pywt.waverec2(denoised_coeffs, 'db4')

            # Trim to original size (waverec2 may pad by 1 pixel)
            reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]

            # Convert back to original range
            return numpy.clip(reconstructed * dtype_max, 0, dtype_max).astype(orig_dtype)

        # Handle grayscale vs color
        if scidata.ndim == 2:
            result = _denoise_channel(scidata)
        else:
            # Denoise each channel independently
            channels = [_denoise_channel(scidata[:, :, c]) for c in range(scidata.shape[2])]
            result = numpy.stack(channels, axis=2)

        logger.info('Applying wavelet denoise (BayesShrink), levels=%d scale=%.2f',
                     levels, scale)

        return result
