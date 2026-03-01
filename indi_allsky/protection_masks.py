"""Utilities for computing spatial protection masks.

This module contains the routines for detecting and masking bright stars,
diffuse Milky‑Way nebulosity, and fine structural detail.  The computation
uses photutils (DAOStarFinder for point sources and Background2D for
extended background) and includes a small LRU cache plus asynchronous
helpers to avoid recomputing masks on frames that have already been
processed.

The masks are returned as float32 arrays in the range [0,1].  Photutils is a
required dependency; attempting to import this module without it will raise
an ImportError.

Example usage::

    import cv2
    from indi_allsky.protection_masks import star_mask, nebula_mask, detail_mask

    img = cv2.imread('frame.tif', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    s = star_mask(img)
    n = nebula_mask(img, star_m=s)
    d = detail_mask(img)

"""

import cv2
import hashlib
import numpy as np
import functools
from concurrent.futures import ThreadPoolExecutor

# photutils imports; this package must be installed.
from astropy.convolution import Gaussian2DKernel
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground

__all__ = [
    "star_mask",
    "nebula_mask",
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


def nebula_mask(img: np.ndarray, star_m: np.ndarray | None = None, percentile: float = 60.0, **pu_kwargs) -> np.ndarray:
    """Return a mask marking diffuse Milky‑Way nebulosity in ``img``.

    Parameters
    ----------
    img : ndarray
        Input grayscale image.
    star_m : ndarray or None
        Optional star mask (as returned by :func:`star_mask`).  If provided
        the star pixels are subtracted from the nebula map so that bright
        cores do not pollute the diffuse mask.
    percentile : float
        Percentile threshold applied to a heavy Gaussian blur to isolate
        extended low-frequency structure.  Lower values produce larger
        nebula regions; adjust as needed.
    """

    # float32 is sufficient for Background2D and halves memory vs float64
    data = img if img.dtype == np.float32 else img.astype(np.float32)
    box_size = pu_kwargs.get('box_size', 128)
    bkg = Background2D(data, box_size, filter_size=pu_kwargs.get('filter_size', (101,101)),
                       bkg_estimator=MedianBackground())
    mask = (bkg.background > np.percentile(bkg.background, percentile)).astype(np.float32)
    if star_m is not None:
        mask = np.clip(mask - star_m, 0.0, 1.0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _morph_kernel_5x5)
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
                     nebula_percentile: float = 60.0, **denoise_kwargs) -> np.ndarray:
    """Denoise ``img`` while preserving stars and nebulae.

    Computes star and nebula masks, applies the ``denoiser`` callable,
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
    nebula_percentile : float
        Passed to :func:`nebula_mask`.
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
    n_m = nebula_mask(gray, star_m=s_m, percentile=nebula_percentile)
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
