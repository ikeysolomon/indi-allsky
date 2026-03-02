"""Utilities for computing star and Milky-Way nebulosity masks.

This module contains the routines for detecting and masking bright stars
and diffuse Milky‑Way nebulosity.  The computation uses photutils
(DAOStarFinder for point sources and Background2D for extended
background) and includes a small LRU cache plus asynchronous helpers to
avoid recomputing masks on frames that have already been processed.

The masks are returned as float32 arrays in the range [0,1].  Photutils is a
required dependency; attempting to import this module without it will raise
an ImportError.

Example usage::

    import cv2
    from indi_allsky.protection_masks import star_mask, nebula_mask

    img = cv2.imread('frame.tif', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    s = star_mask(img)
    n = nebula_mask(img, star_m=s)

"""

import cv2
import numpy as np
import functools
from concurrent.futures import ThreadPoolExecutor

# photutils imports; this package must be installed.
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from astropy.convolution import Gaussian2DKernel

# ``protect_denoiser`` lives in :mod:`denoise`; we expose a thin wrapper
# here to preserve the old public API without introducing a circular import.
__all__ = [
    "star_mask",
    "nebula_mask",
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
    # Use robust sigma estimate to avoid bright sources inflating the std
    try:
        from astropy.stats import sigma_clipped_stats
        _, _, bkg_std = sigma_clipped_stats(data, sigma=3.0)
    except Exception:
        bkg_std = float(np.std(data))

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * bkg_std)
    tbl = daofind(data)
    mask = np.zeros(shape, np.float32)
    if tbl is not None:
        # Create a normalized Gaussian stamp matching the requested FWHM
        stddev = fwhm / 2.3548
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
            # Accumulate using element-wise maximum so overlaps combine
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1],
                                             stamp[sy0:sy1, sx0:sx1])

    # Return inverted mask: 1.0 = sky (unprotected), 0.0 = protected (stars)
    return np.clip(1.0 - mask, 0.0, 1.0).astype(np.float32)

_executor = ThreadPoolExecutor(max_workers=2)

def set_cache_size(size: int):
    """Adjust the LRU star‑mask cache capacity.

    Clearing the cache when the size changes.
    """
    global _cache_max_entries, _cached_star
    _cache_max_entries = size
    _cached_star.cache_clear()
    _cached_star = functools.lru_cache(maxsize=size)(_cached_star)


def async_star_mask(img: np.ndarray, percentile: float = 99.0, threshold_sigma: float = 2.0, fwhm: float = 5.0):
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
    sig = pu_kwargs.get('threshold_sigma', 2.0)
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

    data = img.astype(np.float64)
    box_size = pu_kwargs.get('box_size', 128)
    bkg = Background2D(data, box_size, filter_size=pu_kwargs.get('filter_size', (101,101)),
                       bkg_estimator=MedianBackground())
    mask = (bkg.background > np.percentile(bkg.background, percentile)).astype(np.float32)
    if star_m is not None:
        mask = np.clip(mask - star_m, 0.0, 1.0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def protect_denoiser(img: np.ndarray, denoiser, star_percentile: float = 99.0, nebula_percentile: float = 60.0, **denoise_kwargs) -> np.ndarray:
    """Proxy to :func:`indi_allsky.denoise.protect_denoiser`.

    This lives here solely for compatibility; the real implementation is in
    ``denoise.py`` but importing it at module load time would create a
    circular dependency (``denoise`` needs the mask functions).  The import is
    therefore performed inside the function body.
    """
    from .denoise import protect_denoiser as _pd
    return _pd(img, denoiser, star_percentile=star_percentile, nebula_percentile=nebula_percentile, **denoise_kwargs)
