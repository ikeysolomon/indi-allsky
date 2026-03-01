"""Utilities for computing spatial protection masks.

This module contains the routines for detecting and masking bright stars,
the Milky Way band, and fine structural detail.  The computation
uses photutils (DAOStarFinder for point sources and Background2D for
extended background) and includes a small LRU cache plus asynchronous
helpers to avoid recomputing masks on frames that have already been
processed.

The masks are returned as float32 arrays in the range [0,1].  Photutils is a
required dependency; attempting to import this module without it will raise
an ImportError.

Example usage::

    import cv2
    from indi_allsky.protection_masks import star_mask, milkyway_mask, detail_mask

    img = cv2.imread('frame.tif', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    s = star_mask(img)
    n = milkyway_mask(img, star_m=s)
    d = detail_mask(img)

"""

import cv2
import hashlib
import numpy as np
import functools
from concurrent.futures import ThreadPoolExecutor

from scipy.ndimage import median_filter as _scipy_median_filter

# photutils imports; this package must be installed.
from astropy.convolution import Gaussian2DKernel
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground

__all__ = [
    "star_mask",
    "milkyway_mask",
    "detail_mask",
    "set_cache_size",
    "async_star_mask",
    "protect_denoiser",
]

_morph_kernel_5x5 = np.ones((5, 5), np.uint8)

_cache_max_entries = 8

# LRU cache of star masks keyed by a hash digest plus parameters.
# Using an MD5 hash of the image bytes (16 bytes) rather than the raw
# image bytes (~8 MB for 1920×1080 float32) makes cache key comparison
# essentially free while keeping collision risk negligible.
@functools.lru_cache(maxsize=_cache_max_entries)
def _cached_star(key_hash: bytes, percentile: float, threshold_sigma: float = 2.5, fwhm: float = 3.0, shape=None, _raw_bytes=None):
    # _raw_bytes carries the actual image data; key_hash is just for cache lookup.
    # ``percentile`` is not used in the computation — it only serves as a
    # cache-key discriminator so callers with different percentile values
    # get independent cache slots.  (Leftover from an older Laplacian-based
    # detection approach; kept for API/cache compatibility.)
    data = np.frombuffer(_raw_bytes, dtype=np.float32).reshape(shape)
    # Use sigma-clipped statistics for a robust background σ estimate.
    # Plain np.std(data) is biased high by bright stars, inflating the
    # detection threshold and causing faint stars to be missed.  A 3-σ
    # iterative clip removes stellar outliers, giving the true sky noise.
    _, _, bkg_std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * bkg_std)
    tbl = daofind(data)
    mask = np.zeros(shape, np.float32)
    if tbl is not None:
        # Stamp a Gaussian PSF at each detected star instead of a hard
        # binary circle.  This produces a soft-edged protection mask that
        # tapers realistically with the stellar point-spread function,
        # matching the Gaussian profile that DAOStarFinder itself fits.
        # stddev = fwhm / (2 * sqrt(2 * ln2)) ≈ fwhm / 2.3548
        stddev = fwhm / 2.3548
        # Kernel radius: 3× the FWHM gives >99.7 % of the flux
        radius = int(np.ceil(fwhm * 3))
        kernel = Gaussian2DKernel(x_stddev=stddev, x_size=2 * radius + 1,
                                  y_size=2 * radius + 1)
        stamp = (kernel.array / kernel.array.max()).astype(np.float32)
        sh, sw = stamp.shape
        hh, hw = sh // 2, sw // 2
        h, w = shape
        for x, y in zip(tbl['xcentroid'], tbl['ycentroid']):
            cx, cy = int(round(x)), int(round(y))
            # Compute overlap between stamp and image bounds
            y0 = max(cy - hh, 0)
            y1 = min(cy + hh + 1, h)
            x0 = max(cx - hw, 0)
            x1 = min(cx + hw + 1, w)
            sy0 = y0 - (cy - hh)
            sy1 = sh - ((cy + hh + 1) - y1)
            sx0 = x0 - (cx - hw)
            sx1 = sw - ((cx + hw + 1) - x1)
            # Element-wise maximum so overlapping stars accumulate correctly
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1],
                                             stamp[sy0:sy1, sx0:sx1])
    return mask

_executor = ThreadPoolExecutor(max_workers=2)

def set_cache_size(size: int):
    """Adjust the LRU star‑mask cache capacity.

    Clearing the cache when the size changes.
    """
    global _cache_max_entries, _cached_star
    _cache_max_entries = size
    _cached_star.cache_clear()
    _cached_star = functools.lru_cache(maxsize=size)(_cached_star)


def async_star_mask(img: np.ndarray, percentile: float = 99.0, threshold_sigma: float = 2.5, fwhm: float = 3.0):
    """Return a future computing ``star_mask(img,...)``."""
    return _executor.submit(star_mask, img, percentile=percentile,
                             threshold_sigma=threshold_sigma, fwhm=fwhm)


def star_mask(img: np.ndarray, percentile: float = 99.0, **pu_kwargs) -> np.ndarray:
    """Return a mask marking bright stars in ``img``.

    Parameters
    ----------
    img : ndarray
        Grayscale image (float or uint) normalized to whatever range it uses.
    percentile : float
        Cache-key discriminator inherited from an older Laplacian-based
        approach.  Not used in the current DAOStarFinder detection but
        retained for API/cache compatibility.

    Additional keyword arguments are forwarded to :func:`_cached_star`:
      * ``threshold_sigma`` – detection threshold (in multiples of the
        sigma-clipped background σ).  Defaults to **2.5** (previously 5.0).
      * ``fwhm`` – expected stellar FWHM in pixels (defaults to 3.0).
    """

    # encode image as bytes; hash for cheap cache key comparison
    f32 = img if img.dtype == np.float32 else img.astype(np.float32)
    raw_bytes = f32.tobytes()
    key_hash = hashlib.md5(raw_bytes).digest()
    shape = img.shape
    sig = pu_kwargs.get('threshold_sigma', 2.5)
    fwhm = pu_kwargs.get('fwhm', 3.0)
    # LRU cache call — key_hash is the cache key; raw_bytes passed for data access
    return _cached_star(key_hash, percentile, sig, fwhm, shape, _raw_bytes=raw_bytes)


def milkyway_mask(img: np.ndarray, star_m: np.ndarray | None = None, percentile: float = 60.0, **pu_kwargs) -> np.ndarray:
    """Return a **soft graduated band-following** mask marking the Milky Way.

    Unlike the star mask (which stamps Gaussian PSFs at point sources),
    the Milky Way mask captures extended low-frequency emission — the
    Milky Way band, bright nebulae, zodiacal light, etc.  The returned
    mask is a smooth float32 array in [0, 1] where brighter or more
    central regions get stronger protection and the edges taper gradually.

    Algorithm  (ridge-following)
    ----------------------------
    1. Estimate a low-frequency background model via ``Background2D``
       (photutils) with ``MedianBackground``.
    2. **Detect the ridge** — for each image row, find the column with
       peak (smoothed) background brightness.  Smooth the resulting
       column trace with a median filter → Gaussian blur to produce a
       stable ridge line that tracks the Milky Way band axis.
    3. **Per-row signal strength** — compute how much brighter the ridge
       is compared to the local dark sky (low percentile of the row).
       Rows where the band is weak get proportionally weaker protection.
    4. **Distance-from-ridge falloff** — build a Gaussian profile
       centred on the ridge with a half-width proportional to signal
       strength.  Combine this geometric weight with the actual
       brightness excess to produce the final soft mask.
    5. Apply gamma, Gaussian edge-softening blur, and subtract the star
       mask so point-source protection does not double-count.

    This replaces an older global-percentile-floor approach which
    produced V-shaped masks when the Milky Way crossed the frame
    diagonally (the global floor clipped faint parts of the band while
    allowing bright parts to balloon out).

    Parameters
    ----------
    img : ndarray
        Input grayscale image (float32 preferred).
    star_m : ndarray or None
        Star mask (always computed when denoising is active).  When
        provided, star pixels are subtracted from the Milky Way mask
        so bright stars are not mis-identified as nebulosity.  Accepts
        ``None`` only for standalone / testing use.
    percentile : float
        Controls sensitivity.  Mapped internally to a per-row floor
        percentile: ``row_floor = percentile × 0.25``.  Lower values
        extend protection further into dim sky; higher values restrict
        it to only the brightest emission.  Default 60 → row floor at
        the 15th percentile.

    Keyword arguments (via ``**pu_kwargs``):
      * ``box_size`` – tile size in pixels for ``Background2D`` (default 128).
      * ``filter_size`` – shape of the median filter applied to the
        *background mesh* (not the image).  Must be small relative to
        the mesh dimensions.  Default ``(5, 5)``.
      * ``soft_sigma`` – Gaussian sigma (in pixels) for the edge-softening
        blur applied to the final mask.  Defaults to ``box_size × 0.75``.
      * ``gamma`` – power exponent applied to the normalised ramp before
        blurring.  Values < 1 widen the protected region; values > 1
        concentrate protection on the brightest cores.  Default 0.8.
      * ``band_halfwidth`` – base half-width of the protection band as a
        fraction of the larger image dimension.  Default 0.15 (15 %).
    """

    # float32 is sufficient for Background2D and halves memory vs float64
    data = img if img.dtype == np.float32 else img.astype(np.float32)
    h, w = data.shape[:2]
    box_size = pu_kwargs.get('box_size', 128)

    bkg = Background2D(data, box_size,
                       filter_size=pu_kwargs.get('filter_size', (5, 5)),
                       bkg_estimator=MedianBackground())
    bg = bkg.background.astype(np.float32)

    # ---- 1. Detect the ridge (brightest column per row) ----
    # Light Gaussian blur avoids noise-spike peaks
    bg_smooth = cv2.GaussianBlur(bg, (0, 0), w * 0.05)
    peak_cols = np.argmax(bg_smooth, axis=1).astype(np.float64)

    # Robust smooth: median filter strips outlier rows, then Gaussian
    med_size = min(h // 4 * 2 + 1, 501)  # always odd, ≤ 501
    if med_size < 3:
        med_size = 3
    ridge = _scipy_median_filter(peak_cols, size=med_size)
    ridge = cv2.GaussianBlur(
        ridge.reshape(-1, 1).astype(np.float32), (0, 0), h * 0.1
    ).ravel()

    # ---- 2. Per-row signal strength ----
    # Row floor: low percentile captures the dark sky beside the band
    row_floor_pct = max(float(percentile) * 0.25, 5.0)
    row_floor = np.percentile(bg, row_floor_pct, axis=1).astype(np.float32)

    # Signal = brightness at ridge − row floor
    ridge_idx = np.clip(ridge.astype(np.intp), 0, w - 1)
    row_peak = bg[np.arange(h), ridge_idx]
    row_strength = np.maximum(row_peak - row_floor, 0.0)

    # Normalise to [0, 1]
    strength_95 = np.percentile(row_strength[row_strength > 0], 95) \
        if np.any(row_strength > 0) else 1.0
    if strength_95 < 1e-6:
        return np.zeros(data.shape[:2], dtype=np.float32)
    row_strength_norm = np.clip(row_strength / np.float32(strength_95), 0, 1)

    # ---- 3. Distance-from-ridge Gaussian mask ----
    base_halfwidth = max(h, w) * float(pu_kwargs.get('band_halfwidth', 0.15))
    # Rows with stronger signal get wider protection (50 %–100 % of base)
    row_halfwidth = base_halfwidth * (0.5 + 0.5 * row_strength_norm)

    col_grid = np.arange(w, dtype=np.float32)[np.newaxis, :]  # (1, W)
    ridge_2d = ridge[:, np.newaxis].astype(np.float32)          # (H, 1)
    dist = np.abs(col_grid - ridge_2d)                          # (H, W)

    sigma_hw = (row_halfwidth * 0.6)[:, np.newaxis]             # (H, 1)
    gauss_mask = np.exp(
        np.float32(-0.5) * (dist / np.maximum(sigma_hw, np.float32(1.0))) ** 2
    ).astype(np.float32)

    # Weight by row signal strength
    gauss_mask *= row_strength_norm[:, np.newaxis]

    # ---- 4. Combine with brightness excess ----
    bg_excess = bg - row_floor[:, np.newaxis]
    bg_excess_norm = np.clip(
        bg_excess / np.maximum(np.float32(strength_95), np.float32(1.0)), 0, 1
    )

    # Geometric mean: both ridge-proximity and actual brightness required
    mask = np.sqrt(gauss_mask * bg_excess_norm).astype(np.float32)

    # Renormalise so peak → 1
    mask_peak = np.percentile(mask[mask > 0], 99) if np.any(mask > 0) else 1.0
    if mask_peak > 1e-6:
        mask = np.clip(mask / np.float32(mask_peak), 0, 1)

    # ---- 5. Gamma + soft blur ----
    gamma = float(pu_kwargs.get('gamma', 0.8))
    if abs(gamma - 1.0) > 1e-3:
        np.power(mask, gamma, out=mask)

    soft_sigma = float(pu_kwargs.get('soft_sigma', box_size * 0.75))
    if soft_sigma > 0.5:
        mask = cv2.GaussianBlur(mask, (0, 0), soft_sigma)

    # ---- 6. Subtract star mask ----
    if star_m is not None:
        np.subtract(mask, star_m, out=mask)
        np.clip(mask, 0.0, 1.0, out=mask)

    return mask


def detail_mask(lum: np.ndarray, threshold: float | None = None,
                dtype_max: float = 1.0, lap_multiplier: float = 3.0,
                bright_gate_frac: float = 0.05) -> np.ndarray:
    """Compute a conservative detail mask marking structural edges.

    Returns a boolean HxW array — ``True`` where significant edge detail
    exists *and* the pixel is bright enough to be worth protecting.
    The Laplacian is used to quantify local contrast; pixels whose
    absolute Laplacian exceeds a dynamically-computed threshold *and*
    that are above a configurable brightness gate are flagged.

    Parameters
    ----------
    lum : ndarray
        Float32 luminance image (single-channel).
    threshold : float or None
        Optional brightness threshold used for gating.  If *None* the gate
        is derived from ``dtype_max``.
    dtype_max : float
        Maximum possible value for the image dtype (e.g. 255.0 for uint8).
    lap_multiplier : float
        Factor applied to the median absolute Laplacian to set the detail
        detection threshold.  Higher values are more conservative.
    bright_gate_frac : float
        Fraction of ``threshold`` (or ``dtype_max``) used as the minimum
        brightness to retain.
    """
    try:
        lap = cv2.Laplacian(lum, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        lap_median = float(np.median(lap_abs))
        detail_thresh = max(lap_median * float(lap_multiplier),
                            float(dtype_max) * 0.0005)

        # Brightness gate: prefer using provided threshold; otherwise use
        # a small fraction of the dtype range to avoid preserving noise.
        if threshold is None:
            bright_gate = float(dtype_max) * float(bright_gate_frac)
        else:
            bright_gate = max(float(threshold) * float(bright_gate_frac),
                              float(dtype_max) * 0.005)

        bright_pixels = lum > bright_gate
        return (lap_abs > detail_thresh) & bright_pixels
    except Exception:
        return np.zeros_like(lum, dtype=bool)


def protect_denoiser(img: np.ndarray, denoiser, star_percentile: float = 99.0,
                     milkyway_percentile: float = 60.0, **denoise_kwargs) -> np.ndarray:
    """Denoise ``img`` while preserving stars and the Milky Way.

    Computes star and Milky Way masks, applies the ``denoiser`` callable,
    then blends the protected regions back to the original.  This is a
    standalone convenience wrapper that does not require an
    :class:`~indi_allsky.denoise.IndiAllskyDenoise` instance.

    Parameters
    ----------
    img : ndarray
        Input image (BGR colour or single-channel, any dtype).
    denoiser : callable
        ``denoiser(img, **denoise_kwargs) -> ndarray``.
    star_percentile : float
        Passed to :func:`star_mask`.
    milkyway_percentile : float
        Passed to :func:`milkyway_mask`.
    """
    # Compute luminance for mask generation (cv2 SIMD path)
    if img.ndim == 3 and img.shape[2] >= 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(img, code)
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32)
    else:
        gray = img if img.dtype == np.float32 else img.astype(np.float32)

    # Build joint protection mask
    s_m = star_mask(gray, percentile=star_percentile)
    n_m = milkyway_mask(gray, star_m=s_m, percentile=milkyway_percentile)
    protection = np.maximum(s_m, n_m)

    # Run the user-supplied denoiser
    denoised = denoiser(img, **denoise_kwargs)

    # Blend: mask=1 keeps original, mask=0 keeps denoised
    if protection.max() <= 0.01:
        return denoised

    if img.ndim == 3:
        pm = protection[:, :, np.newaxis]
    else:
        pm = protection

    if np.issubdtype(img.dtype, np.integer):
        dtype_max = float(np.iinfo(img.dtype).max)
    else:
        dtype_max = 1.0

    orig_f = img if img.dtype == np.float32 else img.astype(np.float32)
    den_f = denoised if denoised.dtype == np.float32 else denoised.astype(np.float32)
    result = pm * orig_f + (1.0 - pm) * den_f
    return np.clip(result, 0, dtype_max).astype(img.dtype)
