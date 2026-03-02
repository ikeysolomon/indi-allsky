"""Utilities for computing star protection masks.

This module contains routines for detecting and masking bright stars. The
star mask is produced using photutils' DAOStarFinder and returned as a
float32 array in the range [0, 1] where 1.0 = sky (unprotected) and 0.0 =
protected (star core). A small LRU cache and async helpers are provided to
avoid recomputing masks for identical frames.

Example usage::

    import cv2
    from indi_allsky.protection_masks import star_mask

    img = cv2.imread('frame.tif', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    s = star_mask(img)

"""

import cv2
import numpy as np
import functools
import time
from concurrent.futures import ThreadPoolExecutor

# photutils imports; this package must be installed.
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.convolution import Gaussian2DKernel

# ``protect_denoiser`` lives in :mod:`denoise`; we expose a thin wrapper
# here to preserve the old public API without introducing a circular import.
__all__ = [
    "star_mask",
    "fast_star_mask",
    "set_cache_size",
    "async_star_mask",
    "protect_denoiser",
]

_cache_max_entries = 8

# LRU cache of star masks keyed by image bytes plus parameters
@functools.lru_cache(maxsize=_cache_max_entries)
def _cached_star(key: bytes, percentile: float, threshold_sigma: float, fwhm: float, shape):
    # key is raw bytes; shape is needed to reshape back
    data = np.frombuffer(key, dtype=np.float32).reshape(shape)
    # profiling timers (micro-optimized regions)
    t_start = time.perf_counter()
    prof_sigma = prof_dao = prof_stamp = 0.0
    # Use robust sigma estimate to avoid bright sources inflating the std
    try:
        from astropy.stats import sigma_clipped_stats
        t0 = time.perf_counter()
        _, _, bkg_std = sigma_clipped_stats(data, sigma=3.0)
        prof_sigma = time.perf_counter() - t0
    except Exception:
        t0 = time.perf_counter()
        bkg_std = float(np.std(data))
        prof_sigma = time.perf_counter() - t0

    t0 = time.perf_counter()
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * bkg_std)
    tbl = daofind(data)
    prof_dao = time.perf_counter() - t0

    mask = np.zeros(shape, np.float32)
    if tbl is not None and len(tbl) > 0:
        # Vectorized stamping: build impulse image and convolve with stamp
        stamp = _make_stamp(fwhm) if '_make_stamp' in globals() else None
        if stamp is None:
            stddev = fwhm / 2.3548
            radius = int(np.ceil(fwhm * 3))
            kernel = Gaussian2DKernel(x_stddev=stddev, x_size=2 * radius + 1,
                                      y_size=2 * radius + 1)
            stamp = (kernel.array / kernel.array.max()).astype(np.float32)

        # Build impulses (1 at rounded centroid locations)
        impulses = np.zeros(shape, dtype=np.float32)
        xs = np.rint(tbl['xcentroid']).astype(int)
        ys = np.rint(tbl['ycentroid']).astype(int)
        # Clip to image bounds
        xs = np.clip(xs, 0, shape[1] - 1)
        ys = np.clip(ys, 0, shape[0] - 1)
        impulses[ys, xs] = 1.0

        t1 = time.perf_counter()
        # Convolve impulses with stamp to paint stars; use OpenCV for speed
        # cv2.filter2D will sum overlapping stamps; we'll clip later
        mask = cv2.filter2D(impulses, -1, stamp, borderType=cv2.BORDER_CONSTANT)
        prof_stamp = time.perf_counter() - t1

    # profiling record for diagnostics (cached result of this call)
    t_end = time.perf_counter()
    try:
        _last_profile['sigma_time'] = prof_sigma
        _last_profile['dao_time'] = prof_dao
        _last_profile['stamp_time'] = prof_stamp
        _last_profile['total_time'] = t_end - t_start
        _last_profile['n_stars'] = int(len(tbl)) if tbl is not None else 0
    except Exception:
        pass

    # Return inverted mask: 1.0 = sky (unprotected), 0.0 = protected (stars)
    return np.clip(1.0 - mask, 0.0, 1.0).astype(np.float32)

_executor = ThreadPoolExecutor(max_workers=2)

# Cache Gaussian stamp kernels by FWHM to avoid rebuilding per-star
_kernel_cache: dict[float, np.ndarray] = {}
_last_profile: dict = {}

def get_last_star_profile() -> dict:
    """Return profiling info from the last `_cached_star` invocation.

    Keys: `sigma_time`, `dao_time`, `stamp_time`, `total_time`, `n_stars`.
    """
    return dict(_last_profile)


def set_cache_size(size: int):
    """Adjust the LRU star‑mask cache capacity.

    Clearing the cache when the size changes.
    """
    global _cache_max_entries, _cached_star
    _cache_max_entries = size
    _cached_star.cache_clear()
    _cached_star = functools.lru_cache(maxsize=size)(_cached_star)


def async_star_mask(img: np.ndarray, percentile: float = 99.0, threshold_sigma: float = 1.5, fwhm: float = 5.0):
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
        Pixels whose Laplacian value is above this percentile are marked as
        containing stars.  The default of 99%% is a reasonable starting point
        but can be adjusted based on image characteristics.
    """

    # encode image bytes plus parameters to lookup
    key_bytes = img.astype(np.float32).tobytes()
    shape = img.shape
    sig = pu_kwargs.get('threshold_sigma', 1.5)
    fwhm = pu_kwargs.get('fwhm', 5.0)
    # LRU cache call
    return _cached_star(key_bytes, percentile, sig, fwhm, shape)


def _tile_median_background(img: np.ndarray, box_size: int = 128, filter_size=(5, 5)) -> np.ndarray:
    """Compute a coarse tiled median background and upsample to image size.

    Divides the image into tiles of ``box_size`` and computes each tile's
    median, then resizes and lightly blurs to produce a smooth background
    map matching ``img``'s shape.  This lightweight helper is used for
    visualisation and ridge-tracing in the PR test harness.
    """
    data = img if img.dtype == np.float32 else img.astype(np.float32)
    h, w = data.shape
    nbh = (h + box_size - 1) // box_size
    nbc = (w + box_size - 1) // box_size
    pad_h = nbh * box_size - h
    pad_w = nbc * box_size - w
    if pad_h or pad_w:
        data_p = np.pad(data, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        data_p = data

    # compute medians per tile
    med = np.zeros((nbh, nbc), dtype=np.float32)
    for i in range(nbh):
        for j in range(nbc):
            y0 = i * box_size
            x0 = j * box_size
            med[i, j] = float(np.median(data_p[y0:y0 + box_size, x0:x0 + box_size]))

    # Upsample and crop
    med_up = cv2.resize(med, (nbc * box_size, nbh * box_size), interpolation=cv2.INTER_LINEAR)
    med_up = med_up[:h, :w]

    kx, ky = filter_size if filter_size is not None else (3, 3)
    kx = max(1, int(kx))
    ky = max(1, int(ky))
    bg = cv2.blur(med_up, (kx, ky))
    return bg.astype(np.float32)
# Note: Nebula / Milky-Way nebulosity masking has been removed from the
# public API. If needed in future, implement a separate module with a more
# controlled set of parameters and optional auto-enable logic.


def protect_denoiser(img: np.ndarray, denoiser, star_percentile: float = 99.0, nebula_percentile: float = 60.0, **denoise_kwargs) -> np.ndarray:
    """Proxy to :func:`indi_allsky.denoise.protect_denoiser`.

    This lives here solely for compatibility; the real implementation is in
    ``denoise.py`` but importing it at module load time would create a
    circular dependency (``denoise`` needs the mask functions).  The import is
    therefore performed inside the function body.
    """
    from .denoise import protect_denoiser as _pd
    return _pd(img, denoiser, star_percentile=star_percentile, nebula_percentile=nebula_percentile, **denoise_kwargs)


def _make_stamp(fwhm: float) -> np.ndarray:
    """Return a normalized float32 Gaussian stamp for `fwhm` (cached)."""
    key = float(fwhm)
    if key in _kernel_cache:
        return _kernel_cache[key]
    stddev = fwhm / 2.3548
    radius = int(np.ceil(fwhm * 3))
    kernel = Gaussian2DKernel(x_stddev=stddev, x_size=2 * radius + 1,
                              y_size=2 * radius + 1)
    stamp = (kernel.array / kernel.array.max()).astype(np.float32)
    _kernel_cache[key] = stamp
    return stamp


def fast_star_mask(img: np.ndarray, downsample: int = 4, patch_size: int = 32,
                   percentile: float = 99.0, threshold_sigma: float = 2.0,
                   fwhm: float = 5.0, max_patches: int = 2000) -> np.ndarray:
    """Fast two-stage star detection: coarse candidate selection on a
    downsampled Laplacian, then refine with `DAOStarFinder` on small
    full-resolution patches.

    Returns the same mask semantics as `star_mask` (float32 in [0,1],
    1.0 = sky/unprotected, 0.0 = protected).
    """
    # Ensure grayscale float32
    data = img.astype(np.float32) if img.dtype != np.float32 else img
    if data.ndim == 3 and data.shape[2] >= 3:
        # convert to luminance
        data = (0.299 * data[:, :, 2] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 0]).astype(np.float32)

    h, w = data.shape
    ds = max(1, int(downsample))
    hds = max(1, h // ds)
    wds = max(1, w // ds)

    # Downsample for cheap candidate detection
    small = cv2.resize(data, (wds, hds), interpolation=cv2.INTER_AREA)

    # Laplacian highlights point sources; find local maxima above percentile
    lap = cv2.Laplacian(small, cv2.CV_32F, ksize=3)
    # threshold at requested percentile on the downsampled Laplacian
    try:
        thr = float(np.percentile(lap, float(percentile)))
    except Exception:
        thr = float(np.max(lap))

    # local maxima via simple dilation
    kern3 = np.ones((3, 3), dtype=np.float32)
    dil = cv2.dilate(lap, kern3)
    candidates = (lap == dil) & (lap > thr)

    ys, xs = np.nonzero(candidates)
    if len(xs) == 0:
        return np.clip(np.ones((h, w), dtype=np.float32), 0.0, 1.0)

    # Limit candidate count
    if len(xs) > max_patches:
        idx = np.linspace(0, len(xs) - 1, max_patches).astype(int)
        ys = ys[idx]
        xs = xs[idx]

    # Convert downsample coords to full-res centers
    centers = [(int(x * ds + ds // 2), int(y * ds + ds // 2)) for y, x in zip(ys, xs)]

    mask = np.zeros((h, w), np.float32)
    stamp = _make_stamp(fwhm)
    sh, sw = stamp.shape
    hh, hw = sh // 2, sw // 2

    # Estimate background std on downsampled image for thresholding
    try:
        _, _, bkg_std = sigma_clipped_stats(small, sigma=3.0)
    except Exception:
        bkg_std = float(np.std(small)) + 1e-9
    dao_thresh = threshold_sigma * bkg_std

    daofind = DAOStarFinder(fwhm=fwhm, threshold=dao_thresh)

    for cx, cy in centers:
        x0 = max(cx - patch_size // 2, 0)
        y0 = max(cy - patch_size // 2, 0)
        x1 = min(cx + patch_size // 2 + 1, w)
        y1 = min(cy + patch_size // 2 + 1, h)
        patch = data[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        try:
            tbl = daofind(patch)
        except Exception:
            tbl = None
        if tbl is None:
            continue
        # stamp each detection, translating coords to image space
        for xcent, ycent in zip(tbl['xcentroid'], tbl['ycentroid']):
            gx = int(round(x0 + xcent))
            gy = int(round(y0 + ycent))
            y0s = max(gy - hh, 0)
            y1s = min(gy + hh + 1, h)
            x0s = max(gx - hw, 0)
            x1s = min(gx + hw + 1, w)
            sy0 = y0s - (gy - hh)
            sy1 = sh - ((gy + hh + 1) - y1s)
            sx0 = x0s - (gx - hw)
            sx1 = sw - ((gx + hw + 1) - x1s)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            mask[y0s:y1s, x0s:x1s] = np.maximum(mask[y0s:y1s, x0s:x1s], stamp[sy0:sy1, sx0:sx1])

    return np.clip(1.0 - mask, 0.0, 1.0).astype(np.float32)
