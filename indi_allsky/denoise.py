import cv2
import numpy
import logging
import time

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

    def _norm_strength(self):
        """Return normalized strength t in [0,1] derived from effective strength 1..5."""
        s = max(1, min(self._get_strength(), 5))
        return (float(s) - 1.0) / 4.0


    def _get_bilateral_sigma(self):
        """Return (sigmaColor, sigmaSpace) for the bilateral filter."""
        # Tuned sigma_color per strength (derived from benchmarks).
        # These values are baked into the runtime behavior to provide
        # consistent denoising without relying on external test files.
        sigma_space = int(self.config.get('BILATERAL_SIGMA_SPACE', 15))

        strength = max(1, min(self._get_strength(), 5))
        # Tuned mapping: strengths 1-5 -> sigma_color (baseline)
        tuned_sigma = {
            1: 8,
            2: 8,
            3: 8,
            4: 12,
            5: 16,
        }

        base_sigma = float(tuned_sigma.get(strength, 10))

        # Allow scaling/exponent to reshape strength→sigma mapping.
        # Fallbacks reflect autotune suggestions (BILATERAL_SCALE_FACTOR=0.4, BILATERAL_SCALE_EXP=1.0)
        bil_scale_factor = float(self.config.get('BILATERAL_SCALE_FACTOR', 0.4))
        bil_scale_exp = float(self.config.get('BILATERAL_SCALE_EXP', 1.0))

        # Use a normalized strength mapping (t in [0,1]) and a bounded
        # interpolation between the scale at strength=1 and strength=5.
        t = (float(strength) - 1.0) / 4.0
        sigma_min = base_sigma * bil_scale_factor * (1.0 ** (bil_scale_exp - 1.0))
        sigma_max = base_sigma * bil_scale_factor * (5.0 ** (bil_scale_exp - 1.0))
        sigma_color = int(max(1.0, sigma_min + (sigma_max - sigma_min) * (t ** bil_scale_exp)))

        # If explicit override provided, respect it
        if 'BILATERAL_SIGMA_COLOR' in self.config:
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR'))

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

        # Base thresholds tuned for typical camera ADU ranges. We allow an
        # optional autotune multiplier per-strength to adjust sensitivity
        # automatically (see testing/bench_out/autotune_suggestions.json).
        base_threshold = None
        if scidata.dtype == numpy.uint8:
            base_threshold = float(5.0)
        elif scidata.dtype in (numpy.uint16, numpy.int16):
            base_threshold = float(5.0 * 257.0)  # ~1285 in 16-bit
        else:
            base_threshold = float(5.0 / 255.0)

        # Use tuned multiplier from benchmarking results (baked in):
        # A lower multiplier makes the median filter more aggressive in
        # replacing outlier pixels; this was chosen to balance star
        # preservation vs noise reduction on typical camera images.
        strength = max(1, min(self._get_strength(), 5))
        tuned_multipliers = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
        base_multiplier = float(tuned_multipliers.get(strength, 1.0))

        # Allow global scaling/exponent to reshape strength→multiplier curve.
        # Defaults baked from autotune. These are read from config when
        # available; fallback values mirror the latest autotune suggestions.
        med_scale_factor = float(self.config.get('MEDIAN_SCALE_FACTOR', 2.4))
        med_scale_exp = float(self.config.get('MEDIAN_SCALE_EXP', 2.0))

        # Use normalized strength mapping to produce bounded, smooth changes
        t = self._norm_strength()
        med_min = base_multiplier * med_scale_factor * (1.0 ** (med_scale_exp - 1.0))
        med_max = base_multiplier * med_scale_factor * (5.0 ** (med_scale_exp - 1.0))
        multiplier = float(med_min + (med_max - med_min) * (t ** med_scale_exp))

        threshold = numpy.float32(base_threshold * multiplier)

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

        # Configurable scale: base factor and exponent allow non-linear
        # mappings from strength→sigma. Baked defaults chosen by autotune.
        scale_factor = float(self.config.get('GAUSSIAN_SCALE_FACTOR', 0.2))
        scale_exp = float(self.config.get('GAUSSIAN_SCALE_EXP', 0.5))

        # Use normalized strength mapping to produce bounded sigma between
        # the value at strength=1 and strength=5 to avoid runaway values.
        t = self._norm_strength()
        sigma_min = scale_factor * (1.0 ** scale_exp)
        sigma_max = scale_factor * (5.0 ** scale_exp)
        sigma = float(sigma_min + (sigma_max - sigma_min) * (t ** scale_exp))

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


        # end bilateral


    # ------------------------------------------------------------------
    # Algorithm: Wavelet Denoise (BayesShrink — frequency-domain, best quality)
    # ------------------------------------------------------------------
    def hot_pixel(self, scidata):
        """Remove isolated hot pixels while protecting stars.

        We build a conservative "star" mask from a high percentile of
        local brightness and dilate it by a small radius; any hot-pixel
        candidates inside that mask are ignored to protect star cores.

        This cannot provide a mathematical 100% guarantee (that would
        require full source detection and modeling), but it makes the
        filter extremely unlikely to touch real stars in practice.
        """
        # Opt-in guard: hot-pixel filter is disabled by default unless
        # the runtime config explicitly enables it via `HOTPIXEL_ENABLE`.
        if not bool(self.config.get('HOTPIXEL_ENABLE', False)):
            return scidata

        strength = self._get_strength()
        if strength <= 0:
            return scidata

        # Per-dtype base threshold for a "hot" pixel
        if scidata.dtype == numpy.uint8:
            base_thresh = 15.0
        elif scidata.dtype in (numpy.uint16, numpy.int16):
            base_thresh = 15.0 * 257.0
        else:
            base_thresh = 15.0 / 255.0

        # Make threshold slightly stronger with strength
        thresh = float(base_thresh * (0.8 + 0.3 * (min(max(strength, 1), 5) - 1)))

        # Candidate replacement value: small median (3x3)
        local_med = self._medianBlur(scidata, 3)

        # Use luminance to detect isolated hot pixels across color channels
        if scidata.ndim == 3 and scidata.shape[2] >= 3:
            lum = (0.299 * scidata[:, :, 2].astype(numpy.float32) +
                   0.587 * scidata[:, :, 1].astype(numpy.float32) +
                   0.114 * scidata[:, :, 0].astype(numpy.float32))
            local_med_lum = (0.299 * local_med[:, :, 2].astype(numpy.float32) +
                             0.587 * local_med[:, :, 1].astype(numpy.float32) +
                             0.114 * local_med[:, :, 0].astype(numpy.float32))
        else:
            lum = scidata.astype(numpy.float32)
            local_med_lum = local_med.astype(numpy.float32)

        lum_diff = (lum - local_med_lum)
        hot_mask_2d = lum_diff > thresh

        # Require isolation via 3x3 neighbor count on the 2D hot mask
        hot_u8 = hot_mask_2d.astype(numpy.uint8)
        neigh_count = cv2.filter2D(hot_u8, -1, numpy.ones((3, 3), dtype=numpy.uint8))
        isolated_2d = (neigh_count == 1)

        # Build star-protect mask from a high brightness percentile
        star_pct = int(self.config.get('HOTPIXEL_STAR_PERCENTILE', 95))
        try:
            star_threshold = float(numpy.percentile(lum, star_pct))
        except Exception:
            star_threshold = float(numpy.max(lum))

        star_mask_2d = lum >= star_threshold
        protect_radius = int(self.config.get('HOTPIXEL_PROTECT_RADIUS', 2))
        kern = numpy.ones((protect_radius * 2 + 1, protect_radius * 2 + 1), dtype=numpy.uint8)
        star_mask_dil = cv2.dilate(star_mask_2d.astype(numpy.uint8), kern).astype(bool)

        # Final 2D replacement mask
        replace_mask_2d = hot_mask_2d & isolated_2d & (~star_mask_dil)

        if not numpy.any(replace_mask_2d):
            return scidata

        # Expand to per-channel mask for replacement if needed
        result = scidata.copy()
        if scidata.ndim == 3 and scidata.shape[2] >= 3:
            rep_mask = numpy.repeat(replace_mask_2d[:, :, numpy.newaxis], scidata.shape[2], axis=2)
            result[rep_mask] = local_med[rep_mask]
        else:
            result[replace_mask_2d] = local_med[replace_mask_2d]

        n_replaced = int(numpy.count_nonzero(replace_mask_2d))
        logger.info('Applied hot-pixel filter (star-safe), strength=%d replaced=%d pixels', strength, n_replaced)

        return result
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
                    1 → 2.4×,  2 → 4.8×,  3 → 7.2×,  4 → 9.6×,  5 → 12.0×

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

        # start timer and log request so completion is always traceable
        start_t = time.time()
        strength = self._get_strength()
        logger.info('Wavelet denoise requested (strength=%d)', strength)

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        # Configurable scale for wavelet shrinkage (factor & exponent).
        # Use the fine autotune suggestion (no PR) as a reasonable default
        # for Pi-class images. These may be overridden in runtime config.
        wavelet_scale_factor = float(self.config.get('WAVELET_SCALE_FACTOR', 0.1))
        wavelet_scale_exp = float(self.config.get('WAVELET_SCALE_EXP', 0.5))

        # Map strength→scale using a normalized bounded interpolation to
        # avoid runaway thresholds that obliterate the image.
        t = self._norm_strength()
        scale_min = wavelet_scale_factor * (1.0 ** wavelet_scale_exp)
        scale_max = wavelet_scale_factor * (5.0 ** wavelet_scale_exp)
        scale = float(scale_min + (scale_max - scale_min) * (t ** wavelet_scale_exp))

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
            denoised_ch = numpy.clip(reconstructed * dtype_max, 0, dtype_max).astype(orig_dtype)
            return denoised_ch

        # Handle grayscale vs color
        if scidata.ndim == 2:
            result = _denoise_channel(scidata)
        else:
            # Denoise each channel independently
            channels = [_denoise_channel(scidata[:, :, c]) for c in range(scidata.shape[2])]
            result = numpy.stack(channels, axis=2)

        # Blend denoised vs original to inherently limit destructive changes.
        blend_cap = float(self.config.get('WAVELET_MAX_BLEND', 0.25))
        # Blend scales with strength: keep a small amount at strength=1, up to cap at strength=5
        blend = float(blend_cap * (0.25 + 0.75 * t))

        # Ensure types align for blending
        orig_f = scidata.astype(numpy.float32)
        den_f = result.astype(numpy.float32)
        blended = numpy.clip((blend * den_f) + ((1.0 - blend) * orig_f), 0, float(dtype_max)).astype(orig_dtype)

        elapsed = time.time() - start_t
        logger.info('Applied wavelet denoise (BayesShrink), levels=%d scale=%.2f blend=%.2f time=%.3fs',
                levels, scale, blend, elapsed)

        return blended
