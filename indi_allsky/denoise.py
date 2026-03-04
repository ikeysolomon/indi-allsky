"""Denoising utilities for indi-allsky.

This module defines :class:`IndiAllskyDenoise`, a collection of filtering
algorithms.  Star protection is delegated to
:mod:`indi_allsky.protection_masks`, ensuring consistent masking logic
across the codebase.
"""

import cv2
import numpy
import logging
import time
import pywt
import concurrent.futures

from . import constants

from .protection_masks import star_mask

# caches to avoid rebuilding small objects repeatedly
_db4_wavelet = None  # will hold a pywt.Wavelet('db4') instance
_wavelet_level_cache: dict[int,int] = {}  # min_dim -> max_level
_hotpixel_kernel_cache: dict[int,numpy.ndarray] = {}  # radius -> kernel

# Shared thread pool to avoid creating executors on every call.  The
# pool size is deliberately kept small (4 workers) to match the cores on a
# typical Raspberry Pi.
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Hoisted constant maps to avoid reallocating them per-call
_GAUSSIAN_SIGMA_MAP = {1: 1.0, 2: 1.8, 3: 3.0, 4: 4.2, 5: 5.8}
_WAVELET_SCALE_MAP = {1: 1.8, 2: 3.0, 3: 4.6, 4: 7.0, 5: 9.2}
_BILATERAL_TUNED_SIGMA = {1: 8, 2: 8, 3: 8, 4: 12, 5: 16}

# tuning constants for various denoising algorithms; extracted from the
# long series of adjustments sprinkled through the original implementation.
# Keeping them here makes it easier to audit and tweak.

# wavelet strength adjustment:
WAVELET_SCALE_ADJUST = 1.10  # 10% boost to the base scale value

# gaussian modifications (sigma & blend) applied as cumulative multipliers.
# combining them once gives a single constant that is easier to read and reason about.
GAUSSIAN_SIGMA_ADJUST = 0.707625   
GAUSSIAN_BLEND_ADJUST = GAUSSIAN_SIGMA_ADJUST

# median strength adjustments for kernel size and blend.
MEDIAN_KSIZE_ADJUST = 0.851       
MEDIAN_BLEND_ADJUST = MEDIAN_KSIZE_ADJUST    

# bilateral strength adjustments tuning
BILATERAL_BLEND_BUMP = 1.20
BILATERAL_SIGMA_BUMP = 1.20

# use Astropy for robust statistics
from astropy.stats import sigma_clipped_stats


logger = logging.getLogger('indi_allsky')


class IndiAllskyDenoise(object):
    """Lightweight image denoising for allsky cameras.

    Provides four denoising algorithms.  Bilateral, gaussian and
    median filters all run at similar speed on target hardware; wavelet is
    noticeably slower but offers the highest quality.

    Algorithms exposed to callers:
      - gaussian_blur: Direct Gaussian blur with strength-based blending
      - median_blur: Direct median filter with strength-based blending
      - bilateral: Edge-aware bilateral filter (preserves star edges)
      - wavelet: BayesShrink wavelet denoise (frequency-domain, best quality)

    All algorithms apply the filter directly and blend with the original
    at a strength-dependent ratio.  Strength 1 gives subtle smoothing;
    strength 5 produces fully-filtered output (visibly smoother).

    Each algorithm respects a configurable strength parameter (1-5).
    Temporal averaging is handled separately by the stacking system.

    Configuration may also tweak algorithm-specific knobs:
      * GAUSSIAN_SIGMA, GAUSSIAN_BLEND
      * MEDIAN_BLEND
      * BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
      * DENOISE_STAR_* (star protection parameters)
      * DENOISE_DOWNSAMPLE (global integer shrink factor — see wavelet doc)
      * ADAPTIVE_BLEND (enable/disable variance‑based blend adaptivity)
      * LOCAL_STATS_KSIZE (window size for variance map, odd integer >=3)
    """

    def __init__(self, config, night_av):
        self.config = config
        self.night_av = night_av

    def _match_luminance(self, orig, result):
        """Scale `result` to match the mean luminance of `orig` to avoid
        perceived brightening/dimming after denoising. Returns a copy.

        The applied gain is clipped to a small range to avoid introducing
        large global contrast changes.
        """
        try:
            if orig.ndim == 3 and orig.shape[2] >= 3:
                orig_lum = (0.299 * orig[:, :, 2].astype(numpy.float32) +
                            0.587 * orig[:, :, 1].astype(numpy.float32) +
                            0.114 * orig[:, :, 0].astype(numpy.float32))
                res_lum = (0.299 * result[:, :, 2].astype(numpy.float32) +
                           0.587 * result[:, :, 1].astype(numpy.float32) +
                           0.114 * result[:, :, 0].astype(numpy.float32))
            else:
                orig_lum = orig.astype(numpy.float32)
                res_lum = result.astype(numpy.float32)

            orig_mean = float(numpy.mean(orig_lum))
            res_mean = float(numpy.mean(res_lum))

            if res_mean <= 1e-6:
                return result

            gain = orig_mean / (res_mean + 1e-9)

            min_gain = float(self.config.get('DENOISE_MIN_LUM_GAIN', 0.92))
            max_gain = float(self.config.get('DENOISE_MAX_LUM_GAIN', 1.08))
            gain = max(min_gain, min(max_gain, gain))

            res_f = result.astype(numpy.float32) * gain
            dtype_max = float(numpy.iinfo(result.dtype).max) if numpy.issubdtype(result.dtype, numpy.integer) else 1.0
            res_f = numpy.clip(res_f, 0.0, dtype_max).astype(result.dtype)
            return res_f
        except Exception:
            return result

    def _compute_luminance(self, img):
        """Return a float32 luminance image for `img` (shape HxW)."""
        if img.ndim == 3 and img.shape[2] >= 3:
            return (0.299 * img[:, :, 2].astype(numpy.float32) +
                    0.587 * img[:, :, 1].astype(numpy.float32) +
                    0.114 * img[:, :, 0].astype(numpy.float32))
        return img.astype(numpy.float32)

    # ------------------------------------------------------------------
    # downsampling helpers
    # ------------------------------------------------------------------
    def _get_downsample(self, for_wavelet: bool = False) -> int:
        """Return integer downsample factor from config.

        * ``DENOISE_DOWNSAMPLE`` is a general key that applies to all
          algorithms.
        * ``WAVELET_DOWNSAMPLE`` is honoured when ``for_wavelet`` is True
          and takes precedence (for backwards compatibility).
        """
        if for_wavelet and 'WAVELET_DOWNSAMPLE' in self.config:
            return int(self.config['WAVELET_DOWNSAMPLE'])
        return int(self.config.get('DENOISE_DOWNSAMPLE', 1))

    def _maybe_shrink(self, img: numpy.ndarray, factor: int) -> numpy.ndarray:
        """Return a reduced version of ``img`` by ``factor``.

        If ``factor`` &le; 1 the original array is returned.  Reduction uses
        ``cv2.INTER_AREA``; caller must upsample the result later if
        necessary.
        """
        if factor <= 1:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)

    def _local_variance(self, img, ksize=3):
        """Compute a local variance map using a fast OpenCV box blur.

        Uses the identity E[x^2] - (E[x])^2 on the luminance channel and
        relies on `cv2.blur` which is significantly faster than the
        astropy Box2DKernel approach for typical image sizes.
        """
        if ksize % 2 == 0:
            ksize += 1
        lum = self._compute_luminance(img).astype(numpy.float32)
        mean = cv2.blur(lum, (ksize, ksize))
        mean_sq = cv2.blur(lum * lum, (ksize, ksize))
        var = mean_sq - mean * mean
        numpy.clip(var, 0.0, None, out=var)
        return var

    def _star_mask(self, img):
        """Return a soft point-source mask using the protection_masks module.

        This wrapper adapts the mask generator to this class's configuration
        format.  Exceptions are caught and a blank mask returned, preserving
        the previous fault-tolerant behaviour.
        """
        try:
            # the protection_masks.star_mask call expects a grayscale float32 image
            if img.ndim == 3 and img.shape[2] >= 3:
                gray = self._compute_luminance(img)
            else:
                gray = img.astype(numpy.float32)

            # build keyword arguments from any DENOISE_STAR_* keys present in
            # the configuration.  When no key is supplied we simply allow
            # ``protection_masks.star_mask`` to use its own DEFAULT_* values.
            param_map = {
                'DENOISE_STAR_PERCENTILE': 'percentile',
                'DENOISE_STAR_SIGMA': 'threshold_sigma',
                'DENOISE_STAR_FWHM': 'fwhm',
                'DENOISE_STAR_PROTECT_RADIUS': 'expand_radius',
            }
            kwargs: dict[str, object] = {}
            for cfg_key, arg_name in param_map.items():
                if cfg_key in self.config:
                    val = self.config[cfg_key]
                    # numeric parameters are stored as strings in the config
                    # so convert to the appropriate type when forwarding.
                    if arg_name == 'expand_radius':
                        kwargs[arg_name] = int(val)
                    else:
                        kwargs[arg_name] = float(val)

            return star_mask(gray, **kwargs)
        except Exception:
            return numpy.zeros(img.shape[:2], dtype=numpy.float32)


    def _apply_star_protection(self, original, denoised, dtype_max):
        """Blend star regions back to the original, preserving point sources.

        If DENOISE_PROTECT_STARS is False (or the star mask is empty)
        the denoised image is returned unchanged.
        """
        if not bool(self.config.get('DENOISE_PROTECT_STARS', True)):
            return denoised

        # obtain soft star protection mask (delegates to protection_masks.star_mask)
        star_mask = self._star_mask(original)

        if not numpy.any(star_mask > 0.01):
            return denoised

        # Expand to 3D if colour image
        if original.ndim == 3:
            sm = star_mask[:, :, numpy.newaxis]
        else:
            sm = star_mask

        orig_f = original.astype(numpy.float32)
        den_f = denoised.astype(numpy.float32)
        # `star_mask` uses 1.0 = sky (unprotected), 0.0 = protected (stars).
        # We want protected pixels to retain the original and sky to use
        # the denoised value. Therefore blend as:
        #   result = star_mask * denoised + (1 - star_mask) * original
        result = sm * den_f + (1.0 - sm) * orig_f
        return numpy.clip(result, 0, dtype_max).astype(original.dtype)

    def _get_strength(self):
        """Return the effective denoise strength (int) respecting night/day config."""
        if self.config.get('USE_NIGHT_COLOR', True) or self.night_av[constants.NIGHT_NIGHT]:
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
        # These values provide consistent denoising without external test files.
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
        bil_scale_factor = float(self.config.get('BILATERAL_SCALE_FACTOR', 0.4))
        bil_scale_exp = float(self.config.get('BILATERAL_SCALE_EXP', 1.0))

        t = (float(strength) - 1.0) / 4.0
        sigma_min = base_sigma * bil_scale_factor * (1.0 ** (bil_scale_exp - 1.0))
        sigma_max = base_sigma * bil_scale_factor * (5.0 ** (bil_scale_exp - 1.0))
        sigma_color = int(max(1.0, sigma_min + (sigma_max - sigma_min) * (t ** bil_scale_exp)))

        # If explicit override provided, respect it; otherwise bump by 10%
        if 'BILATERAL_SIGMA_COLOR' in self.config:
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR'))
        else:
            # bump computed sigma_color by a fixed factor
            sigma_color = int(max(1.0, sigma_color * BILATERAL_SIGMA_BUMP))

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

        # Fast path for small kernels on small images: skip thread overhead
        if ksize <= 5 and img.shape[0] < 512 and img.shape[1] < 512:
            if img.ndim == 2:
                return _blur_channel(img)
            channels = cv2.split(img)
            blurred = [_blur_channel(ch) for ch in channels]
            return cv2.merge(blurred)

        # Single-channel image
        if img.ndim == 2:
            return _blur_channel(img)

        # Multi-channel: process each channel independently and merge
        channels = cv2.split(img)
        # submit per-channel work to shared thread pool
        futures = [_thread_pool.submit(_blur_channel, ch) for ch in channels]
        blurred = [f.result() for f in futures]
        return cv2.merge(blurred)

    # ------------------------------------------------------------------
    # Algorithm: Median Blur (direct — effective general-purpose denoising)
    # ------------------------------------------------------------------
    def median_blur(self, scidata):
        """Apply a direct median blur blended with the original.

        Applies a median filter at a strength-dependent kernel size and
        blends the result with the original.  The median filter naturally
        preserves edges (unlike Gaussian) while effectively smoothing
        noise.

        Strength mapping:
          1 → 3×3 kernel, blend=0.40   (gentle)
          3 → 7×7 kernel, blend=0.70   (moderate)
          5 → 11×11 kernel, blend=1.00  (strong — fully filtered)

        Strength range: 1-5.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        # optional downsample to save CPU across all algorithms
        down = self._get_downsample()
        img = self._maybe_shrink(scidata, down)

        strength = max(1, min(strength, 5))
        base_ksize = strength * 2 + 1  # always odd: 3,5,7,9,11
        # apply combined kernel-size adjustment constant
        ksize = int(round(base_ksize * MEDIAN_KSIZE_ADJUST))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)

        blurred_small = self._medianBlur(img, ksize)
        if down > 1:
            blurred = cv2.resize(blurred_small,
                                 (scidata.shape[1], scidata.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            blurred = blurred_small

        # Blend fraction: 30% at strength=1 → 80% at strength=5
        t = self._norm_strength()
        base_blend = float(self.config.get('MEDIAN_BLEND', 0.35 + 0.57 * t))
        # apply combined blend adjustment constant
        blend = max(0.0, min(1.0, base_blend * MEDIAN_BLEND_ADJUST))

        # --- adaptive blending using fast local variance map --------------------------------
        # Poisson noise in astro exposures benefits from reducing the
        # blend fraction in regions of high local variance (stars,
        # bright noise) while letting the nominal blend apply in smooth
        # background areas.  A small fixed window variance map costs only
        # two cv2.blur() calls and is thus essentially free.
        adaptive_blend = blend
        if bool(self.config.get('ADAPTIVE_BLEND', True)) and blend > 0.0 and blend < 1.0:
            k = int(self.config.get('LOCAL_STATS_KSIZE', 3))
            var_map = self._local_variance(scidata, k)
            mean_var = float(numpy.mean(var_map)) + 1e-9
            var_norm = numpy.clip(var_map / mean_var, 0.0, 1.0)
            adapt = 1.0 - var_norm
            if scidata.ndim == 3:
                adapt = adapt[:, :, numpy.newaxis]
            adaptive_blend = blend * adapt
        # -------------------------------------------------------------------------------------

        if numpy.issubdtype(scidata.dtype, numpy.integer):
            dtype_max = float(numpy.iinfo(scidata.dtype).max)
        else:
            dtype_max = 1.0

        # blend using adaptive map if available
        result_f32 = adaptive_blend * blurred.astype(numpy.float32) + (1.0 - adaptive_blend) * scidata.astype(numpy.float32)
        result = numpy.clip(result_f32, 0, dtype_max).astype(scidata.dtype)

        # Match luminance to prevent perceived brightening/dimming
        result = self._match_luminance(scidata, result)

        # Protect stars: blend star regions back to original
        result = self._apply_star_protection(scidata, result, dtype_max)

        # log average blend for diagnostics
        avg_blend = float(numpy.mean(adaptive_blend)) if isinstance(adaptive_blend, numpy.ndarray) else adaptive_blend
        logger.info('Applying median denoise, ksize=%d base_blend=%.2f avg_blend=%.2f', ksize, blend, avg_blend)

        return result


    # ------------------------------------------------------------------
    # Algorithm: Gaussian Blur (direct — simple and effective)
    # ------------------------------------------------------------------
    def gaussian_blur(self, scidata):
        # NOTE: any future improvements to gauss/median/bilateral should keep
        # compute cost comparable to the existing implementation.  We try to
        # avoid expensive full‑resolution filtering by threading each channel
        # and by falling back to single‑channel variants when possible.

        """Apply a direct Gaussian blur blended with the original.

        Applies cv2.GaussianBlur at a strength-dependent sigma and
        linearly blends the blurred result with the original.  Higher
        strengths use a larger sigma *and* a larger blend fraction so
        the smoothing effect is always clearly visible.

        Strength mapping (defaults, configurable via GAUSSIAN_SIGMA_MAP):
          1 → σ≈1.5, blend=0.30   (gentle)
          3 → σ≈5.0, blend=0.65   (moderate)
          5 → σ≈11,  blend=1.00   (strong — fully blurred)

        Strength range: 1-5.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        # downsample<br>
        down = self._get_downsample()
        img = self._maybe_shrink(scidata, down)

        strength = max(1, min(strength, 5))

        # Sigma per strength level.  Configurable via GAUSSIAN_SIGMA
        # (overrides the whole map) or GAUSSIAN_SIGMA_MAP (per-level).
        default_sigma_map = {1: 1.0, 2: 1.8, 3: 3.0, 4: 4.2, 5: 5.8}
        sigma = float(self.config.get('GAUSSIAN_SIGMA', default_sigma_map.get(strength, 3.0)))
        # apply global gaussian sigma adjustment
        sigma = sigma * GAUSSIAN_SIGMA_ADJUST

        # perform blur on possibly reduced image
        if img.ndim == 2 or img.shape[2] < 2:
            blurred_small = cv2.GaussianBlur(img, (0, 0), sigma)
        else:
            futures = [_thread_pool.submit(cv2.GaussianBlur, img[:, :, c], (0, 0), sigma)
                       for c in range(img.shape[2])]
            channels = [f.result() for f in futures]
            blurred_small = numpy.stack(channels, axis=2)

        if down > 1:
            blurred = cv2.resize(blurred_small,
                                 (scidata.shape[1], scidata.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            blurred = blurred_small

        # Blend fraction: 20% at strength=1 → 70% at strength=5
        t = self._norm_strength()
        base_blend = float(self.config.get('GAUSSIAN_BLEND', 0.25 + 0.55 * t))
        # apply global gaussian blend adjustment
        blend = max(0.0, min(1.0, base_blend * GAUSSIAN_BLEND_ADJUST))

        # --- adaptive blending using fast local variance map --------------------------------
        adaptive_blend = blend
        if bool(self.config.get('ADAPTIVE_BLEND', True)) and blend > 0.0 and blend < 1.0:
            k = int(self.config.get('LOCAL_STATS_KSIZE', 3))
            var_map = self._local_variance(scidata, k)
            mean_var = float(numpy.mean(var_map)) + 1e-9
            var_norm = numpy.clip(var_map / mean_var, 0.0, 1.0)
            adapt = 1.0 - var_norm
            if scidata.ndim == 3:
                adapt = adapt[:, :, numpy.newaxis]
            adaptive_blend = blend * adapt
        # -------------------------------------------------------------------------------------

        if numpy.issubdtype(scidata.dtype, numpy.integer):
            dtype_max = float(numpy.iinfo(scidata.dtype).max)
        else:
            dtype_max = 1.0

        result_f32 = adaptive_blend * blurred.astype(numpy.float32) + (1.0 - adaptive_blend) * scidata.astype(numpy.float32)
        result = numpy.clip(result_f32, 0, dtype_max).astype(scidata.dtype)

        # Match luminance to prevent perceived brightening/dimming
        result = self._match_luminance(scidata, result)

        # Protect stars: blend star regions back to original
        result = self._apply_star_protection(scidata, result, dtype_max)

        avg_blend = float(numpy.mean(adaptive_blend)) if isinstance(adaptive_blend, numpy.ndarray) else adaptive_blend
        logger.info('Applying gaussian denoise, sigma=%.1f base_blend=%.2f avg_blend=%.2f', sigma, blend, avg_blend)

        return result


    # ------------------------------------------------------------------
    # Algorithm: Bilateral Filter (edge-aware, high quality)
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

        # downsample to accelerate filter
        down = self._get_downsample()
        img = self._maybe_shrink(scidata, down)

        strength = max(1, min(strength, 5))

        d = strength * 2 + 1
        sigma_color, sigma_space = self._get_bilateral_sigma()

        # blend fraction like other algorithms
        t = self._norm_strength()
        blend = float(self.config.get('BILATERAL_BLEND', 0.25 + 0.55 * t))
        # boost blend by fixed constant
        blend = max(0.0, min(1.0, blend * BILATERAL_BLEND_BUMP))

        adaptive_blend = blend
        if bool(self.config.get('ADAPTIVE_BLEND', True)) and 0.0 < blend < 1.0:
            k = int(self.config.get('LOCAL_STATS_KSIZE', 3))
            var_map = self._local_variance(scidata, k)
            mean_var = float(numpy.mean(var_map)) + 1e-9
            var_norm = numpy.clip(var_map / mean_var, 0.0, 1.0)
            adapt = 1.0 - var_norm
            if scidata.ndim == 3:
                adapt = adapt[:, :, numpy.newaxis]
            adaptive_blend = blend * adapt

        needs_conversion = img.dtype not in (numpy.uint8, numpy.float32)
        
        # Compute dtype_max once for use in blend and logging
        if numpy.issubdtype(img.dtype, numpy.integer):
            dtype_max = numpy.float32(numpy.iinfo(img.dtype).max)
        else:
            dtype_max = numpy.float32(1.0)

        # perform filter on downsized image
        if needs_conversion:
            # bilateralFilter supports uint8 and float32.
            # OpenCV float32 bilateral is optimized for 0.0-1.0 range.
            # Normalize data to 0-1, scale sigmaColor to match (divide
            # by 255 since user-facing sigma is calibrated for 0-255).
            # sigmaSpace is in pixel units and needs no adjustment.
            sigma_color_norm = float(sigma_color) / 255.0
            img_f32 = img.astype(numpy.float32) / dtype_max
            denoised_small_f32 = cv2.bilateralFilter(img_f32, d, sigma_color_norm, float(sigma_space))
            denoised_small = numpy.clip(numpy.rint(denoised_small_f32 * dtype_max), 0, float(dtype_max)).astype(img.dtype)
        else:
            denoised_small = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

        if down > 1:
            denoised = cv2.resize(denoised_small,
                                  (scidata.shape[1], scidata.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
        else:
            denoised = denoised_small

        # blend with adaptive map and protect stars
        if numpy.issubdtype(scidata.dtype, numpy.integer):
            dtype_max = float(numpy.iinfo(scidata.dtype).max)
        else:
            dtype_max = 1.0

        result_f32 = adaptive_blend * denoised.astype(numpy.float32) + (1.0 - adaptive_blend) * scidata.astype(numpy.float32)
        result = numpy.clip(result_f32, 0, dtype_max).astype(scidata.dtype)

        # Match luminance to prevent perceived brightening/dimming
        result = self._match_luminance(scidata, result)

        result = self._apply_star_protection(scidata, result, dtype_max)

        avg_blend = float(numpy.mean(adaptive_blend)) if isinstance(adaptive_blend, numpy.ndarray) else adaptive_blend
        log_msg = 'Applying bilateral denoise, d=%d sigmaColor=%d sigmaSpace=%d base_blend=%.2f avg_blend=%.2f'
        if needs_conversion:
            log_msg += ' (float32 converted)'
        logger.info(log_msg, d, sigma_color, sigma_space, blend, avg_blend)

        return result


        # end bilateral

    # ------------------------------------------------------------------
    # Algorithm: Non-local Means (patch-based)
    # ------------------------------------------------------------------
    # Older experiments included non-local means filter, but
    # it proved far too slow on target hardware so the implementation was
    # dropped.  This is a warning.



    # ------------------------------------------------------------------
    # Algorithm: Wavelet Denoise (BayesShrink — frequency-domain, best quality)
    # ------------------------------------------------------------------
    def wavelet(self, scidata):
        """Apply wavelet-based denoising with BayesShrink adaptive thresholding.

        Decomposes the image into frequency bands using the Discrete
        Wavelet Transform (DWT), estimates noise in the finest detail
        coefficients, and shrinks noisy coefficients using BayesShrink
        adaptive soft thresholding.

        The strength parameter controls both the threshold scaling and
        the blend ratio with the original image:
          1 → gentle shrinkage, 40% blend   (subtle smoothing)
          3 → moderate shrinkage, 70% blend  (visible smoothing)
          5 → strong shrinkage, 100% blend   (aggressive, slightly blurry)

        Wavelet: Daubechies-4 (db4), levels: auto (3-4), soft thresholding.
        Requires PyWavelets (pywt).  Strength range: 1-5.
        The denoiser may optionally run on a downsampled copy of the image to
        save CPU; set config key ``WAVELET_DOWNSAMPLE`` to an integer factor
        (&gt;1) for this behaviour.  The result is upscaled before blending and
        star protection, providing a tunable quality/speed trade-off.
        """
        start_t = time.time()
        strength = self._get_strength()
        logger.info('Wavelet denoise requested (strength=%d)', strength)

        if strength <= 0:
            return scidata

        # Optional pre‑downsampling to reduce work.  On constrained hardware
        # such as a Pi a factor‑2 reduction trades a small amount of detail for
        # roughly a 3× speedup in the core wavelet loop.  The denoised result is
        # up‑sampled back to the original size before blending and star
        # protection.  The factor may be specified either via
        # ``WAVELET_DOWNSAMPLE`` (old key) or the new general
        # ``DENOISE_DOWNSAMPLE``; the helper handles precedence.
        down = self._get_downsample(for_wavelet=True)
        if down > 1:
            h, w = scidata.shape[:2]
            small = cv2.resize(scidata, (w // down, h // down), interpolation=cv2.INTER_LINEAR)
        else:
            small = scidata

        strength = max(1, min(strength, 5))

        # Simple linear scale: strength 1→1.5x, 3→4.0x, 5→8.0x
        # These directly multiply the BayesShrink threshold.
        default_scale_map = {1: 1.8, 2: 3.0, 3: 4.6, 4: 7.0, 5: 9.2}
        scale = float(self.config.get('WAVELET_SCALE', default_scale_map.get(strength, 4.6)))
        # apply the simplified single adjustment constant (≈1.10)
        scale = float(scale) * WAVELET_SCALE_ADJUST

        # Determine dtype range for normalization; use the downsampled image
        # when computing levels etc, since we may be working on `small`.
        target = small
        if numpy.issubdtype(target.dtype, numpy.integer):
            dtype_max = float(numpy.iinfo(target.dtype).max)
        else:
            dtype_max = 1.0

        orig_dtype = target.dtype

        # Auto decomposition levels: 3-4 based on smallest image dimension
        # if we downsampled use the smaller dimensions for level computation
        min_dim = min(target.shape[0], target.shape[1])
        # cache wavelet object and level computation
        global _db4_wavelet
        if _db4_wavelet is None:
            _db4_wavelet = pywt.Wavelet('db4')
        if min_dim in _wavelet_level_cache:
            max_level = _wavelet_level_cache[min_dim]
        else:
            max_level = pywt.dwt_max_level(min_dim, _db4_wavelet.dec_len)
            _wavelet_level_cache[min_dim] = max_level
        levels = min(max(max_level, 1), 4)

        def _denoise_channel(channel):
            """Denoise a single 2D channel using BayesShrink."""
            # Normalize to 0-1 float for wavelet precision (float32 is adequate
            # and significantly faster than float64 on large arrays).
            data = channel.astype(numpy.float32) / dtype_max

            # Forward DWT
            coeffs = pywt.wavedec2(data, 'db4', level=levels)

            # Estimate noise sigma from finest detail coefficients (HH band).
            # Replace the heavier astropy call with a simple MAD estimator using
            # pure numpy; this avoids importing and calling astropy for every
            # channel (and saves about 10–15 % of the total runtime).
            detail_hh = coeffs[-1][2]  # HH = diagonal detail at finest level
            # median absolute deviation → Gaussian sigma
            med = numpy.median(detail_hh)
            mad = numpy.median(numpy.abs(detail_hh - med))
            sigma_noise = mad / 0.6745

            # Enforce a minimum noise floor so the filter always does
            # *something* even on high-SNR / well-lit images.
            min_sigma = float(self.config.get('WAVELET_MIN_SIGMA', 0.005))
            sigma_noise = max(sigma_noise, min_sigma)

            # Apply BayesShrink to each detail level.  The loops have been
            # flattened/expressed as comprehensions wherever possible to reduce
            # Python overhead; bands are small so the savings are modest but
            # measurable.  We also compute the variance once per band.
            denoised_coeffs = [coeffs[0]]  # keep approximation untouched
            for detail_level in coeffs[1:]:
                new_details = []
                for detail_band in detail_level:
                    # compute band variance once
                    var_band = numpy.var(detail_band)
                    sigma_band = numpy.sqrt(max(var_band - sigma_noise * sigma_noise, 0.0))
                    if sigma_band < 1e-10:
                        threshold = numpy.max(numpy.abs(detail_band))
                    else:
                        threshold = (sigma_noise * sigma_noise) / sigma_band
                    threshold *= scale
                    new_details.append(pywt.threshold(detail_band, threshold, mode='soft'))
                denoised_coeffs.append(tuple(new_details))

            # Inverse DWT
            reconstructed = pywt.waverec2(denoised_coeffs, 'db4')
            reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]

            return numpy.clip(reconstructed * dtype_max, 0, dtype_max).astype(orig_dtype)

        # Handle grayscale vs color on the *target* image (small or full).
        if target.ndim == 2:
            result_small = _denoise_channel(target)
        else:
            futures = [_thread_pool.submit(_denoise_channel, target[:, :, c])
                       for c in range(target.shape[2])]
            channels = [f.result() for f in futures]
            result_small = numpy.stack(channels, axis=2)

        # if we downsampled, rescale back to original size before blending
        if down > 1:
            h, w = scidata.shape[:2]
            result = cv2.resize(result_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            result = result_small

        # Blend: 40% at strength=1 → 100% at strength=5
        t = self._norm_strength()
        blend = float(self.config.get('WAVELET_BLEND', 0.46 + 0.54 * t))
        blend = max(0.0, min(1.0, blend))

        orig_f = scidata.astype(numpy.float32)
        den_f = result.astype(numpy.float32)
        blended = numpy.clip((blend * den_f) + ((1.0 - blend) * orig_f), 0, float(dtype_max)).astype(orig_dtype)

        # Match luminance to prevent perceived brightening/dimming
        blended = self._match_luminance(scidata, blended)

        # Protect stars: blend star regions back to original
        blended = self._apply_star_protection(scidata, blended, float(dtype_max))

        elapsed = time.time() - start_t
        logger.info('Applied wavelet denoise (BayesShrink) down=%d levels=%d scale=%.2f blend=%.2f time=%.3fs',
                down, levels, scale, blend, elapsed)

        return blended



