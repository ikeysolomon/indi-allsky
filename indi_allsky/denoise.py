"""Denoising utilities for indi-allsky.

This module defines :class:`IndiAllskyDenoise`, a collection of filtering
algorithms.  Star and Milky Way protection is delegated to
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

from .protection_masks import star_mask as _pm_star
from .protection_masks import milkyway_mask as _pm_mw
from .protection_masks import detail_mask as _pm_detail

# caches to avoid rebuilding small objects repeatedly
_db4_wavelet = pywt.Wavelet('db4')
_wavelet_level_cache: dict[int,int] = {}  # min_dim -> max_level


logger = logging.getLogger('indi_allsky')


class IndiAllskyDenoise:
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

    .. note:: **Hot-pixel removal was intentionally removed.**

       An earlier version included a ``hot_pixel()`` pre-pass that tried to
       detect and replace isolated hot pixels before the main denoise
       algorithm ran.  It used a percentile-based brightness gate + dilation
       to protect stars, but in practice the heuristic could not reliably
       distinguish a genuine faint star from a sensor hot-pixel.  Because
       star preservation is the primary goal of this pipeline, the feature
       was removed to avoid silently destroying real star data.  If a future
       contributor revisits this idea, consider:

         * Using dark-frame subtraction (the proper way to handle hot pixels)
           rather than spatial heuristics on a single light frame.
         * Building an explicit hot-pixel map from dark/bias calibration
           frames and only replacing listed coordinates.
         * The ``_median_blur(scidata, 3)`` replacement value worked well;
           the detection side was the problem.

       The ``darks.py`` module already provides dark-frame support and is the
       recommended path for hot-pixel mitigation.

    Configuration may also tweak algorithm-specific knobs:
      * GAUSSIAN_SIGMA, GAUSSIAN_BLEND
      * MEDIAN_BLEND
      * BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
      * DENOISE_STAR_* (star protection parameters)
      * DENOISE_PROTECT_MILKYWAY (enable Milky Way protection in joint mask)
      * DENOISE_MILKYWAY_BOX_SIZE, DENOISE_MILKYWAY_FILTER_SIZE (Background2D mesh tuning)
      * DENOISE_PROTECT_DETAIL (enable edge/detail protection in joint mask)
      * ADAPTIVE_BLEND (enable/disable variance‑based blend adaptivity)
      * LOCAL_STATS_KSIZE (window size for variance map, odd integer >=3)
    """

    # ------------------------------------------------------------------
    # Class-level thread pool: reused across calls to avoid creating a
    # new ThreadPoolExecutor on every frame.  The worker count is modest
    # since individual tasks (per-channel blur) are short-lived.
    # ------------------------------------------------------------------
    _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # Constant lookup tables — hoisted from per-frame methods to avoid
    # recreating identical small dicts on every single frame.
    _GAUSSIAN_SIGMA_MAP = {1: 1.0, 2: 1.8, 3: 3.0, 4: 4.2, 5: 5.8}
    _WAVELET_SCALE_MAP = {1: 1.8, 2: 3.0, 3: 4.6, 4: 7.0, 5: 9.2}
    _BILATERAL_SIGMA_MAP = {1: 8, 2: 8, 3: 8, 4: 12, 5: 16}

    def __init__(self, config, night_av):
        self.config = config
        self.night_av = night_av

    @staticmethod
    def _dtype_max(arr):
        """Return the maximum representable value for *arr*'s dtype."""
        if numpy.issubdtype(arr.dtype, numpy.integer):
            return float(numpy.iinfo(arr.dtype).max)
        return 1.0

    @staticmethod
    def _as_f32(arr):
        """Return *arr* as float32, avoiding a copy when already float32."""
        return arr if arr.dtype == numpy.float32 else arr.astype(numpy.float32)

    def _adaptive_blend(self, scidata, blend):
        """Return an adaptive blend map or scalar *blend*.

        When ``ADAPTIVE_BLEND`` is enabled and *blend* is in the open
        interval (0, 1), a local-variance map is used to reduce the
        blend fraction in high-variance regions (stars, bright noise)
        while keeping the nominal fraction in smooth sky background.

        Returns either a float scalar (unchanged *blend*) or a float32
        ndarray broadcastable to *scidata*'s shape.
        """
        if not bool(self.config.get('ADAPTIVE_BLEND', True)):
            return blend
        if blend <= 0.0 or blend >= 1.0:
            return blend

        k = int(self.config.get('LOCAL_STATS_KSIZE', 3))
        var_map = self._local_variance(scidata, k)
        mean_var = float(numpy.mean(var_map)) + 1e-9
        var_norm = numpy.clip(var_map / mean_var, 0.0, 1.0)
        adapt = 1.0 - var_norm
        if scidata.ndim == 3:
            adapt = adapt[:, :, numpy.newaxis]
        return blend * adapt

    def _blend_and_protect(self, scidata, filtered, adaptive_blend, log_fmt, *log_args):
        """Common tail shared by median_blur, gaussian_blur, and bilateral.

        Blends *filtered* with *scidata* using *adaptive_blend* (scalar or
        ndarray), corrects luminance drift, applies star/Milky Way protection,
        logs diagnostics, and returns the final result in *scidata*'s dtype.
        """
        dtype_max = self._dtype_max(scidata)

        # y + a*(x-y) form with in-place ops: 1 temp array instead of 3-4.
        filt_f = self._as_f32(filtered)
        orig_f = self._as_f32(scidata)
        result_f32 = numpy.subtract(filt_f, orig_f)
        result_f32 *= adaptive_blend
        result_f32 += orig_f
        numpy.clip(result_f32, 0, dtype_max, out=result_f32)
        result = result_f32.astype(scidata.dtype)

        result = self._match_luminance(scidata, result, dtype_max)
        result = self._apply_protection(scidata, result, dtype_max)

        avg_blend = (float(numpy.mean(adaptive_blend))
                     if isinstance(adaptive_blend, numpy.ndarray)
                     else adaptive_blend)
        logger.info(log_fmt, *log_args, avg_blend)
        return result

    def _match_luminance(self, orig, result, dtype_max=None):
        """Scale `result` to match the mean luminance of `orig` to avoid
        perceived brightening/dimming after denoising. Returns a copy.

        *dtype_max* may be passed to avoid a redundant :meth:`_dtype_max`
        call when the caller already has the value.

        The applied gain is clipped to a small range to avoid introducing
        large global contrast changes.
        """
        try:
            orig_lum = self._compute_luminance(orig)
            res_lum = self._compute_luminance(result)

            orig_mean = float(numpy.mean(orig_lum))
            res_mean = float(numpy.mean(res_lum))

            if res_mean <= 1e-6:
                return result

            gain = orig_mean / (res_mean + 1e-9)

            # Clip gain to avoid extreme global changes but allow modest
            # correction to prevent perceived brightening/dimming.
            min_gain = float(self.config.get('DENOISE_MIN_LUM_GAIN', 0.92))
            max_gain = float(self.config.get('DENOISE_MAX_LUM_GAIN', 1.08))
            gain = max(min_gain, min(max_gain, gain))

            # When gain is effectively unity the multiply is pure overhead.
            if abs(gain - 1.0) < 1e-6:
                return result

            # Apply gain uniformly to all channels
            dmax = dtype_max if dtype_max is not None else self._dtype_max(result)
            res_f = self._as_f32(result) * gain
            res_f = numpy.clip(res_f, 0.0, dmax).astype(result.dtype)
            return res_f
        except Exception:
            return result

    def _compute_luminance(self, img):
        """Return a float32 luminance image for `img` (shape HxW).

        Uses ``cv2.cvtColor`` which applies the same Rec.601 weights as
        the previous hand-rolled weighted sum but via SIMD-optimised C++
        (~3-5× faster on large frames).
        """
        if img.ndim == 3 and img.shape[2] >= 3:
            code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(img, code)
            return self._as_f32(gray)
        return self._as_f32(img)

    def _local_variance(self, img, ksize=3):
        """Compute a local variance map using a fixed square window.

        The variance is computed on the luminance channel using the standard
        E[x^2] - (E[x])^2 identity.  ``ksize`` is forced to an odd integer
        >= 1.  Uses cv2.blur() (box mean) which is orders of magnitude
        faster than astropy convolution for this purpose.
        The returned array is float32, same height/width as ``img``
        (single-channel).
        """
        if ksize % 2 == 0:
            ksize += 1
        lum = self._compute_luminance(img)
        # cv2.blur computes the box-mean — equivalent to convolve with a
        # normalized Box2DKernel but vastly faster (SIMD-optimized C++).
        mean = cv2.blur(lum, (ksize, ksize))
        mean_sq = cv2.blur(lum * lum, (ksize, ksize))
        var = mean_sq - mean * mean
        # negative values can arise from rounding; clamp
        numpy.clip(var, 0.0, None, out=var)
        return var

    def _build_protection_mask(self, img):
        """Build a combined protection mask from stars, Milky Way, and detail.

        Returns a float32 HxW array in [0, 1] where 1 = fully protect
        (keep original) and 0 = allow full denoising.

        The three mask computations (star, Milky Way, detail) operate at
        different spatial scales and are independent of each other, so
        they are dispatched concurrently to the class-level thread pool.
        The Milky Way mask's star-subtraction step is applied *after* both
        futures resolve — this lets star and Milky Way detection overlap in
        time while still producing the correct final mask.

        Steps:
          1. Submit star mask to thread pool (always).
          2. If DENOISE_PROTECT_MILKYWAY, submit Milky Way mask (without
             star subtraction — that happens in the merge step).
          3. If DENOISE_PROTECT_DETAIL, submit detail mask.
          4. Collect results, subtract star mask from Milky Way mask,
             and take element-wise maximum of all masks.
        """
        want_milkyway = bool(self.config.get('DENOISE_PROTECT_MILKYWAY', False))
        want_detail = bool(self.config.get('DENOISE_PROTECT_DETAIL', False))

        # Pre-compute luminance once — the module-level mask functions
        # each need a float32 grayscale image, so calling them directly
        # with pre-computed lum avoids redundant full-frame cv2.cvtColor
        # calls per frame.
        lum = self._compute_luminance(img)
        dmax = self._dtype_max(img)

        # Extract config once for the concurrent submissions
        star_pct = float(self.config.get('DENOISE_STAR_PERCENTILE', 99.0))
        star_sig = float(self.config.get('DENOISE_STAR_SIGMA', 2.5))
        star_fwhm = float(self.config.get('DENOISE_STAR_FWHM', 3.0))

        # --- dispatch independent mask computations concurrently ----------
        star_future = self._thread_pool.submit(
            _pm_star, lum, percentile=star_pct,
            threshold_sigma=star_sig, fwhm=star_fwhm)

        mw_future = None
        if want_milkyway:
            # Run Milky Way detection WITHOUT star_m so it can proceed in
            # parallel.  Star subtraction is applied after both finish.
            mw_pct = float(self.config.get('DENOISE_MILKYWAY_PERCENTILE', 60.0))
            mw_box = int(self.config.get('DENOISE_MILKYWAY_BOX_SIZE', 128))
            mw_fs = int(self.config.get('DENOISE_MILKYWAY_FILTER_SIZE', 5))
            mw_future = self._thread_pool.submit(
                _pm_mw, lum, star_m=None, percentile=mw_pct,
                box_size=mw_box, filter_size=(mw_fs, mw_fs))

        detail_future = None
        if want_detail:
            detail_future = self._thread_pool.submit(_pm_detail, lum, None, dmax)

        # --- collect results and merge ------------------------------------
        star_m = star_future.result()
        protection = star_m

        if mw_future is not None:
            mw_m = mw_future.result()
            # Subtract star pixels from Milky Way mask (same logic that was
            # previously inside milkyway_mask via star_m= parameter).
            mw_m = numpy.clip(mw_m - star_m, 0.0, 1.0)
            protection = numpy.maximum(protection, mw_m)

        if detail_future is not None:
            detail_bool = detail_future.result()
            protection = numpy.maximum(protection, detail_bool.astype(numpy.float32))

        return protection

    def _apply_protection(self, original, denoised, dtype_max):
        """Blend protected regions back to the original, preserving
        point sources (stars) and optionally extended emission (Milky Way).

        If DENOISE_PROTECT_STARS is False the denoised image is returned
        unchanged.  Otherwise the joint star+Milky Way mask gates blending:
        pixels with mask ~ 1 are kept from the original; pixels with
        mask ~ 0 use the denoised value.
        """
        if not bool(self.config.get('DENOISE_PROTECT_STARS', True)):
            return denoised

        # Build the joint protection mask (stars + optional Milky Way)
        protection = self._build_protection_mask(original)

        # Fast exit when nothing is flagged
        if protection.max() <= 0.01:
            return denoised

        # Expand mask to 3D for colour images so broadcasting works
        if original.ndim == 3:
            pm = protection[:, :, numpy.newaxis]
        else:
            pm = protection

        # Weighted blend: protected pixels -> original, rest -> denoised
        # y + a*(x-y) form with in-place ops: 1 temp instead of ~4.
        orig_f = self._as_f32(original)
        den_f = self._as_f32(denoised)
        result = numpy.subtract(orig_f, den_f)
        result *= pm
        result += den_f
        numpy.clip(result, 0, dtype_max, out=result)
        return result.astype(original.dtype)

    def _get_strength(self):
        """Return the effective denoise strength (int) respecting night/day config."""
        if self.config.get('USE_NIGHT_COLOR', True) or self.night_av[constants.NIGHT_NIGHT]:
            return int(self.config.get('IMAGE_DENOISE_STRENGTH', 3))
        return int(self.config.get('IMAGE_DENOISE_STRENGTH_DAY', 3))

    def _norm_strength(self, strength=None):
        """Return normalized strength t in [0,1] derived from effective strength 1..5.

        If *strength* is provided it is used directly, avoiding a
        redundant :meth:`_get_strength` call when the caller already
        has the value.
        """
        if strength is None:
            strength = self._get_strength()
        s = max(1, min(strength, 5))
        return (float(s) - 1.0) / 4.0


    def _get_bilateral_sigma(self, strength=None):
        """Return (sigmaColor, sigmaSpace) for the bilateral filter.

        If *strength* is provided it is used directly, avoiding a
        redundant :meth:`_get_strength` call.
        """
        sigma_space = int(self.config.get('BILATERAL_SIGMA_SPACE', 15))

        if strength is None:
            strength = self._get_strength()
        strength = max(1, min(strength, 5))

        base_sigma = float(self._BILATERAL_SIGMA_MAP.get(strength, 10))

        # Allow scaling/exponent to reshape strength→sigma mapping.
        bil_scale_factor = float(self.config.get('BILATERAL_SCALE_FACTOR', 0.4))
        bil_scale_exp = float(self.config.get('BILATERAL_SCALE_EXP', 1.0))

        t = (float(strength) - 1.0) / 4.0
        # 1.0 ** x is always 1.0, so sigma_min simplifies directly.
        sigma_min = base_sigma * bil_scale_factor
        sigma_max = base_sigma * bil_scale_factor * (5.0 ** (bil_scale_exp - 1.0))
        sigma_color = int(max(1.0, sigma_min + (sigma_max - sigma_min) * (t ** bil_scale_exp)))

        # If explicit override provided, respect it
        if 'BILATERAL_SIGMA_COLOR' in self.config:
            sigma_color = int(self.config.get('BILATERAL_SIGMA_COLOR'))

        return max(1, sigma_color), max(1, sigma_space)


    def _median_blur(self, img, ksize):
        """cv2.medianBlur only supports CV_8U for multi-channel images in newer OpenCV.
        Split into per-channel blurs when the image is 16-bit (or deeper) multi-channel."""
        def _blur_channel(ch):
            # Fast path for 8-bit single-channel
            if ch.dtype == numpy.uint8:
                return cv2.medianBlur(ch, ksize)

            # Integer types (commonly uint16)
            if numpy.issubdtype(ch.dtype, numpy.integer):
                # cv2.medianBlur supports CV_16U directly for ksize ≤ 5,
                # giving full 16-bit precision without lossy quantization.
                if ch.dtype == numpy.uint16 and ksize <= 5:
                    return cv2.medianBlur(ch, ksize)
                # Fallback for larger kernels or other int types: scale
                # down to 8-bit, blur, scale back (lossy but functional).
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
        # Use class-level thread pool to avoid per-call overhead
        futures = [self._thread_pool.submit(_blur_channel, ch) for ch in channels]
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

        strength = max(1, min(strength, 5))
        ksize = strength * 2 + 1  # always odd: 3, 5, 7, 9, 11

        blurred = self._median_blur(scidata, ksize)

        # Blend fraction: 30% at strength=1 → 80% at strength=5
        t = self._norm_strength(strength)
        blend = float(self.config.get('MEDIAN_BLEND', 0.35 + 0.57 * t))
        blend = max(0.0, min(1.0, blend))

        adaptive_blend = self._adaptive_blend(scidata, blend)

        return self._blend_and_protect(
            scidata, blurred, adaptive_blend,
            'Applying median denoise, ksize=%d base_blend=%.2f avg_blend=%.2f',
            ksize, blend)


    # ------------------------------------------------------------------
    # Algorithm: Gaussian Blur (direct — simple and effective)
    # ------------------------------------------------------------------
    def gaussian_blur(self, scidata):
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

        strength = max(1, min(strength, 5))

        # Sigma per strength level.  Configurable via GAUSSIAN_SIGMA
        # (overrides the whole map) or GAUSSIAN_SIGMA_MAP (per-level).
        sigma = float(self.config.get('GAUSSIAN_SIGMA', self._GAUSSIAN_SIGMA_MAP.get(strength, 3.0)))

        # Blend fraction: 20% at strength=1 → 70% at strength=5
        t = self._norm_strength(strength)
        blend = float(self.config.get('GAUSSIAN_BLEND', 0.25 + 0.55 * t))
        blend = max(0.0, min(1.0, blend))

        adaptive_blend = self._adaptive_blend(scidata, blend)

        # cv2.GaussianBlur handles multi-channel images natively with
        # internal SIMD parallelism — no need for per-channel threading.
        blurred = cv2.GaussianBlur(scidata, (0, 0), sigma)

        return self._blend_and_protect(
            scidata, blurred, adaptive_blend,
            'Applying gaussian denoise, sigma=%.1f base_blend=%.2f avg_blend=%.2f',
            sigma, blend)


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

        strength = max(1, min(strength, 5))

        d = strength * 2 + 1
        sigma_color, sigma_space = self._get_bilateral_sigma(strength)

        # blend fraction like other algorithms
        t = self._norm_strength(strength)
        blend = float(self.config.get('BILATERAL_BLEND', 0.25 + 0.55 * t))
        blend = max(0.0, min(1.0, blend))

        adaptive_blend = self._adaptive_blend(scidata, blend)

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

            logger.info('Applying bilateral denoise (float32 0-1), d=%d sigmaColor=%.4f sigmaSpace=%d base_blend=%.2f', d, sigma_color_norm, sigma_space, blend)
            denoised_f32 = cv2.bilateralFilter(scidata_f32, d, sigma_color_norm, float(sigma_space))

            denoised = numpy.clip(numpy.rint(denoised_f32 * dtype_max), 0, float(dtype_max)).astype(scidata.dtype)
        else:
            denoised = cv2.bilateralFilter(scidata, d, sigma_color, sigma_space)

        # blend with adaptive map and protect stars
        return self._blend_and_protect(
            scidata, denoised, adaptive_blend,
            'Applying bilateral denoise, d=%d sigmaColor=%d sigmaSpace=%d base_blend=%.2f avg_blend=%.2f',
            d, sigma_color, sigma_space, blend)

    # ------------------------------------------------------------------
    # Algorithm: Non-local Means (patch-based)
    # ------------------------------------------------------------------
    # Older experiments included non-local means filter, but
    # it proved far too slow on target hardware so the implementation was
    # dropped.

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
        """
        start_t = time.monotonic()
        strength = self._get_strength()
        logger.info('Wavelet denoise requested (strength=%d)', strength)

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        # Simple linear scale: strength 1→1.5x, 3→4.0x, 5→8.0x
        # These directly multiply the BayesShrink threshold.
        scale = float(self.config.get('WAVELET_SCALE', self._WAVELET_SCALE_MAP.get(strength, 4.6)))

        # Determine dtype range for normalization
        dtype_max = self._dtype_max(scidata)
        dtype_max_f32 = numpy.float32(dtype_max)

        orig_dtype = scidata.dtype

        # Auto decomposition levels: 3-4 based on smallest image dimension
        min_dim = min(scidata.shape[0], scidata.shape[1])
        # cache level computation
        if min_dim in _wavelet_level_cache:
            max_level = _wavelet_level_cache[min_dim]
        else:
            max_level = pywt.dwt_max_level(min_dim, _db4_wavelet.dec_len)
            _wavelet_level_cache[min_dim] = max_level
        levels = min(max(max_level, 1), 4)

        # Read config once before dispatching per-channel threads
        min_sigma = float(self.config.get('WAVELET_MIN_SIGMA', 0.005))

        def _denoise_channel(channel):
            """Denoise a single 2D channel using BayesShrink."""
            # Normalize to 0-1 float32 — halves memory vs float64 and
            # PyWavelets handles float32 natively.
            data = channel.astype(numpy.float32) / dtype_max_f32

            # Forward DWT
            coeffs = pywt.wavedec2(data, _db4_wavelet, level=levels)

            # Estimate noise sigma from finest detail coefficients (HH band)
            # Standard BayesShrink MAD estimator: σ = median(|d|) / 0.6745
            # This is the textbook formula and eliminates the astropy
            # dependency that denoise.py previously carried.
            detail_hh = coeffs[-1][2]  # HH = diagonal detail at finest level
            sigma_noise = float(numpy.median(numpy.abs(detail_hh))) / 0.6745

            # Enforce a minimum noise floor so the filter always does
            # *something* even on high-SNR / well-lit images.
            sigma_noise = max(sigma_noise, min_sigma)
            sigma_noise_sq = sigma_noise ** 2

            # Apply BayesShrink to each detail level
            denoised_coeffs = [coeffs[0]]  # keep approximation untouched
            for i in range(1, len(coeffs)):
                new_details = []
                for detail_band in coeffs[i]:
                    sigma_band = numpy.sqrt(max(numpy.var(detail_band) - sigma_noise_sq, 0))
                    if sigma_band < 1e-10:
                        threshold = numpy.max(numpy.abs(detail_band))
                    else:
                        threshold = sigma_noise_sq / sigma_band

                    # Scale by user strength
                    threshold *= scale

                    new_details.append(pywt.threshold(detail_band, threshold, mode='soft'))

                denoised_coeffs.append(tuple(new_details))

            # Inverse DWT
            reconstructed = pywt.waverec2(denoised_coeffs, _db4_wavelet)
            reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]

            return numpy.clip(reconstructed * dtype_max_f32, 0, dtype_max_f32)

        # Handle grayscale vs color — channels are independent so we
        # dispatch them to the class-level thread pool for ~3× speedup
        # on BGR images (wavelet is the most CPU-intensive algorithm).
        if scidata.ndim == 2:
            result = _denoise_channel(scidata)
        else:
            futures = [
                self._thread_pool.submit(_denoise_channel, scidata[:, :, c])
                for c in range(scidata.shape[2])
            ]
            channels = [f.result() for f in futures]
            result = cv2.merge(channels)

        # Blend: 40% at strength=1 → 100% at strength=5
        t = self._norm_strength(strength)
        blend = float(self.config.get('WAVELET_BLEND', 0.46 + 0.54 * t))
        blend = max(0.0, min(1.0, blend))

        orig_f = self._as_f32(scidata)
        # result is already float32 from _denoise_channel
        blended = numpy.subtract(result, orig_f)
        blended *= numpy.float32(blend)
        blended += orig_f
        numpy.clip(blended, 0, dtype_max_f32, out=blended)
        blended = blended.astype(orig_dtype)

        # Correct luminance drift introduced by wavelet reconstruction + blending
        blended = self._match_luminance(scidata, blended, dtype_max)

        # Protect stars (and optionally Milky Way): blend protected regions back
        blended = self._apply_protection(scidata, blended, dtype_max)

        elapsed = time.monotonic() - start_t
        logger.info('Applied wavelet denoise (BayesShrink), levels=%d scale=%.2f blend=%.2f time=%.3fs',
                levels, scale, blend, elapsed)

        return blended
