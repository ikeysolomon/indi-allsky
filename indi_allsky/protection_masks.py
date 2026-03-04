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
import os
import functools
import time
from concurrent.futures import ThreadPoolExecutor

# photutils imports; this package must be installed.
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.convolution import Gaussian2DKernel

# ``protect_denoiser`` lives in :mod:`denoise`; we expose a thin wrapper
# here to preserve the old public API without introducing a circular import.

# DIAL TO TWIDDLE: Star protection expansion in pixels
# Change this value to adjust the size of protection circles around stars
# in the denoised output. Larger values = bigger star protection zones.
# This is the master control for overlay generation.
DEFAULT_EXPAND_RADIUS: int = 4

# Default parameters for star detection
DEFAULT_PERCENTILE: float = 95.0           # Percentile threshold for star detection
DEFAULT_THRESHOLD_SIGMA: float = 2.0        # SNR threshold: pixels N-sigma above background (neighbours)
DEFAULT_FWHM: float = 5.0                   # Assumed star size in pixels (FWHM of PSF)
DEFAULT_THRESHOLD_SIGMA_FAST: float = 2.0   # More aggressive SNR threshold for two-stage fast detection

# Image processing constants
SIGMA_CLIP_SIGMA: float = 3.0               # Sigma clipping threshold for background estimation
FWHM_TO_STDDEV: float = 2.3548              # Conversion factor: FWHM = stddev * 2.3548
FWHM_RADIUS_MULTIPLIER: int = 3             # Stamp radius = ceil(FWHM * multiplier)
PROTECTED_PIXEL_THRESHOLD: float = 0.5      # Threshold for identifying protected pixels (stars)
LAPLACIAN_KSIZE: int = 3                    # Kernel size for Laplacian edge detection
DISTANCE_TRANSFORM_KSIZE: int = 5           # Kernel size for distance transform

__all__ = [
    "star_mask",
    "fast_star_mask",
    "set_cache_size",
    "async_star_mask",
    ]

_cache_max_entries = 8

# LRU cache of star masks keyed by image bytes plus parameters
@functools.lru_cache(maxsize=_cache_max_entries)
def _cached_star(key: bytes, percentile: float, threshold_sigma: float, fwhm: float, expand_radius: int | None, shape):
    # key is raw bytes; shape is needed to reshape back
    data = np.frombuffer(key, dtype=np.float32).reshape(shape)
    # profiling timers (micro-optimized regions)
    t_start = time.perf_counter()
    prof_sigma = prof_dao = prof_stamp = 0.0
    # Use robust sigma estimate to avoid bright sources inflating the std
    try:
        from astropy.stats import sigma_clipped_stats
        t0 = time.perf_counter()
        _, _, bkg_std = sigma_clipped_stats(data, sigma=SIGMA_CLIP_SIGMA)
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
            stddev = fwhm / FWHM_TO_STDDEV
            radius = int(np.ceil(fwhm * FWHM_RADIUS_MULTIPLIER))
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
        _last_profile['expand_radius'] = int(expand_radius) if expand_radius is not None else 0
    except Exception:
        pass

    # Return inverted mask: 1.0 = sky (unprotected), 0.0 = protected (stars)
    out = np.clip(1.0 - mask, 0.0, 1.0).astype(np.float32)

    # Apply star protection dilation: controlled by DEFAULT_EXPAND_RADIUS (line 35).
    # This expands the protected circle per-star to reduce denoising artifacts.
    try:
        if expand_radius and int(expand_radius) > 0:
            r = int(expand_radius)
            prot_bin = (out < PROTECTED_PIXEL_THRESHOLD).astype(np.uint8)
            h, w = prot_bin.shape
            max_r = max(h, w)
            if r > max_r:
                r = max_r
            # Use distance-transform for all radii (consistent, efficient)
            src = ((prot_bin == 0).astype(np.uint8) * 255)
            dt = cv2.distanceTransform(src, cv2.DIST_L2, DISTANCE_TRANSFORM_KSIZE)
            dil = (dt <= float(r)).astype(np.uint8)
            out[dil.astype(bool)] = 0.0
    except Exception:
        pass

    return out

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

# Async helper to compute star mask without blocking caller; returns a Future. FWHM and other parameters can be passed via kwargs. See `star_mask` for details.
def async_star_mask(img: np.ndarray, percentile: float = DEFAULT_PERCENTILE, threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA, fwhm: float = DEFAULT_FWHM, expand_radius: int = 0):
    """Return a future computing ``star_mask(img,...)``."""
    return _executor.submit(star_mask, img, percentile=percentile,
                             threshold_sigma=threshold_sigma, fwhm=fwhm, expand_radius=expand_radius)


def star_mask(img: np.ndarray, percentile: float = DEFAULT_PERCENTILE, expand_radius: int | None = None, **pu_kwargs) -> np.ndarray:
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

    # if caller didn't provide a radius, use module default
    if expand_radius is None:
        expand_radius = DEFAULT_EXPAND_RADIUS

    # encode image bytes plus parameters to lookup
    key_bytes = img.astype(np.float32).tobytes()
    shape = img.shape
    sig = pu_kwargs.get('threshold_sigma', DEFAULT_THRESHOLD_SIGMA)
    fwhm = pu_kwargs.get('fwhm', DEFAULT_FWHM)
    # LRU cache call (include expand_radius in cache key)
    return _cached_star(key_bytes, percentile, sig, fwhm, int(expand_radius), shape)


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

# Note: Nebula / Milky-Way nebulosity masking was experimented with but removed. 
# If needed in future, implement a separate module due to the extreme number of variables needed. 


def _make_stamp(fwhm: float) -> np.ndarray:
    """Return a normalized float32 Gaussian stamp for `fwhm` (cached)."""
    key = float(fwhm)
    if key in _kernel_cache:
        return _kernel_cache[key]
    stddev = fwhm / FWHM_TO_STDDEV
    radius = int(np.ceil(fwhm * FWHM_RADIUS_MULTIPLIER))
    kernel = Gaussian2DKernel(x_stddev=stddev, x_size=2 * radius + 1,
                              y_size=2 * radius + 1)
    stamp = (kernel.array / kernel.array.max()).astype(np.float32)
    _kernel_cache[key] = stamp
    return stamp


def fast_star_mask(img: np.ndarray, downsample: int = 4, patch_size: int = 32,
                   percentile: float = DEFAULT_PERCENTILE, threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA_FAST,
                   fwhm: float = DEFAULT_FWHM, max_patches: int = 2000, expand_radius: int | None = None) -> np.ndarray:
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
    lap = cv2.Laplacian(small, cv2.CV_32F, ksize=LAPLACIAN_KSIZE)
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
        _, _, bkg_std = sigma_clipped_stats(small, sigma=SIGMA_CLIP_SIGMA)
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

    out = np.clip(1.0 - mask, 0.0, 1.0).astype(np.float32)

    # Apply star protection dilation: controlled by DEFAULT_EXPAND_RADIUS (line 35).
    # This expands the protected circle per-star to reduce denoising artifacts.
    try:
        if expand_radius is None:
            expand_radius = DEFAULT_EXPAND_RADIUS
        if expand_radius and int(expand_radius) > 0:
            r = int(expand_radius)
            prot_bin = (out < PROTECTED_PIXEL_THRESHOLD).astype(np.uint8)
            h, w = prot_bin.shape
            max_r = max(h, w)
            if r > max_r:
                r = max_r
            # use distance transform for all radii to avoid magic numbers
            src = ((prot_bin == 0).astype(np.uint8) * 255)
            dt = cv2.distanceTransform(src, cv2.DIST_L2, DISTANCE_TRANSFORM_KSIZE)
            dil = (dt <= float(r)).astype(np.uint8)
            out[dil.astype(bool)] = 0.0
    except Exception:
        pass

    return out
