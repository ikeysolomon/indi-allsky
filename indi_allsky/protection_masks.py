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
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * np.std(data))
    tbl = daofind(data)
    mask = np.zeros(shape, np.float32)
    if tbl is not None:
        for x, y in zip(tbl['xcentroid'], tbl['ycentroid']):
            cv2.circle(mask, (int(x), int(y)), int(fwhm), 1, -1)
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


def async_star_mask(img: np.ndarray, percentile: float = 99.0, threshold_sigma: float = 5.0, fwhm: float = 3.0):
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
    sig = pu_kwargs.get('threshold_sigma', 5.0)
    fwhm = pu_kwargs.get('fwhm', 3.0)
    # LRU cache call
    return _cached_star(key_bytes, percentile, sig, fwhm, shape)


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
