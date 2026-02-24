import cv2
import numpy
import time
import logging

from numpy.lib.stride_tricks import sliding_window_view

from . import constants


logger = logging.getLogger('indi_allsky')


class IndiAllskyDenoise(object):
    """Lightweight image denoising for allsky cameras.

    Provides four denoising algorithms (quality order):
      - gaussian_blur: Fast Gaussian filter (gentle smoothing)
      - median_blur: Fast median filter (removes hot pixels / salt-and-pepper noise)
      - bilateral: Edge-aware filter (smooths sky while preserving star edges)
      - pca_nlm: PCA-accelerated Non-Local Means (best quality, higher cost)

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


    # ------------------------------------------------------------------
    # Algorithm: PCA-accelerated Non-Local Means (best quality)
    # ------------------------------------------------------------------
    def pca_nlm(self, scidata):
        """Apply PCA-accelerated Non-Local Means denoising.

        Uses patch-based similarity like classic NLM but reduces each patch
        to a small number of PCA components before computing distances.
        This preserves NLM's ability to recognise repeated textures and
        structures while cutting the cost of the distance computation.

        The strength parameter controls both patch size and filter strength:
          patch_size = strength * 2 + 1   (3, 5, 7, 9, 11)
          search_radius = strength + 2    (3, 4, 5, 6, 7)
          h (filter weight) = strength * 0.02

        For colour images, only the luminance (Y) channel is denoised —
        chrominance carries little perceptible noise and this gives a 3×
        speed-up.  The vectorised shift-and-compare approach processes the
        full image per shift, avoiding slow per-pixel loops.

        Typical Pi 4 performance at strength 3: ~1.5-2.5 s.
        Strength range: 1-5.
        """
        strength = self._get_strength()

        if strength <= 0:
            return scidata

        strength = max(1, min(strength, 5))

        patch_radius = strength
        patch_size = 2 * patch_radius + 1
        search_radius = strength + 2
        n_components = min(patch_size * patch_size, 8)
        h = strength * 0.02  # filter strength in 0-1 intensity space

        t_start = time.monotonic()

        # --- Normalise to 0-1 float32 ---------------------------------
        original_dtype = scidata.dtype
        if original_dtype == numpy.uint8:
            dtype_max = numpy.float32(255.0)
        elif numpy.issubdtype(original_dtype, numpy.integer):
            dtype_max = numpy.float32(numpy.iinfo(original_dtype).max)
        else:
            dtype_max = numpy.float32(1.0)

        work = scidata.astype(numpy.float32) / dtype_max

        # --- Colour: denoise luminance only (3× faster) ----------------
        if work.ndim == 3:
            # BGR → YCrCb; denoise Y, leave Cr/Cb untouched
            ycrcb = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = self._pca_nlm_channel(
                ycrcb[:, :, 0], patch_radius, search_radius,
                n_components, h,
            )
            work = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            work = self._pca_nlm_channel(
                work, patch_radius, search_radius, n_components, h,
            )

        # --- Convert back to original dtype ----------------------------
        result = numpy.clip(numpy.rint(work * dtype_max), 0, float(dtype_max))
        result = result.astype(original_dtype)

        elapsed = time.monotonic() - t_start
        logger.info(
            'PCA-NLM denoise done in %.2f s  (patch=%d search=%d pca=%d h=%.3f)',
            elapsed, patch_size, search_radius * 2 + 1, n_components, h,
        )
        return result


    def _pca_nlm_channel(self, channel, patch_radius, search_radius,
                         n_components, h):
        """PCA-NLM on a single 2-D float32 channel in 0-1 range.

        Uses the vectorised shift-and-compare approach: for every possible
        displacement (dx, dy) inside the search window, compute the
        PCA-space distance for *all* pixels at once, derive NLM weights,
        and accumulate the weighted average.  No per-pixel Python loop.
        """
        H, W = channel.shape
        pr = patch_radius
        ps = 2 * pr + 1  # patch side length
        sr = search_radius
        h2 = numpy.float32(h * h)

        # Pad for patch extraction (reflect avoids edge artefacts)
        padded = numpy.pad(channel, pr, mode='reflect')

        # Extract overlapping patches — zero-copy via stride tricks
        # Shape: (H, W, ps, ps)
        patches = sliding_window_view(padded, (ps, ps))[:H, :W]
        # Flatten each patch to a vector: (H, W, ps*ps)
        patch_dim = ps * ps
        patches_flat = patches.reshape(H, W, patch_dim)

        # --- PCA basis from a random subsample -------------------------
        n_pixels = H * W
        n_samples = min(n_pixels, 20000)
        rng = numpy.random.default_rng(42)  # deterministic for consistency
        idx = rng.choice(n_pixels, n_samples, replace=False)
        sample = patches_flat.reshape(-1, patch_dim)[idx]

        mean_patch = sample.mean(axis=0, dtype=numpy.float32)
        centred = sample - mean_patch

        # Economy SVD — only need first n_components right-singular vectors
        _, _, Vt = numpy.linalg.svd(centred, full_matrices=False)
        basis = Vt[:n_components].astype(numpy.float32)  # (n_comp, patch_dim)

        # Project every patch into PCA space: (H, W, n_comp)
        pca_all = (patches_flat - mean_patch) @ basis.T

        # --- Pad PCA volume and original channel for shifting ----------
        pca_padded = numpy.pad(
            pca_all,
            ((sr, sr), (sr, sr), (0, 0)),
            mode='reflect',
        )
        chan_padded = numpy.pad(channel, sr, mode='reflect')

        # --- Shift-and-compare NLM ------------------------------------
        weight_sum = numpy.zeros((H, W), dtype=numpy.float32)
        result = numpy.zeros((H, W), dtype=numpy.float32)

        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                # PCA patch of the shifted neighbour
                shifted_pca = pca_padded[
                    sr + dy: sr + dy + H,
                    sr + dx: sr + dx + W,
                    :,
                ]
                # Squared Euclidean distance in PCA space
                diff = pca_all - shifted_pca
                dist2 = numpy.einsum('hwc,hwc->hw', diff, diff)

                # NLM weight  (exp clipped to avoid underflow)
                w = numpy.exp(numpy.clip(-dist2 / h2, -80.0, 0.0))

                # Accumulate weighted original pixel values
                shifted_px = chan_padded[
                    sr + dy: sr + dy + H,
                    sr + dx: sr + dx + W,
                ]
                weight_sum += w
                result += w * shifted_px

        # Normalise
        result /= numpy.maximum(weight_sum, numpy.float32(1e-10))
        return result
