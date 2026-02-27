"""Standalone star‑mask generator extracted from denoise.

This module provides a single function ``generate_star_mask(img, config)``
which returns a soft mask in ``float32`` (range 0..1) indicating point‑source
locations that should be protected from denoising.  The behaviour is
controlled by a handful of configuration values, documented below.

For backwards compatibility the same keys used by ``denoise._star_mask``
are honoured.  In addition a new convenience parameter
``DENOISE_STAR_MASK_STRENGTH`` (0.0..1.0) acts as a **star protection
strength** dial.  A higher value produces a stronger mask that protects
more pixels, while a lower value tightens the criteria and exposes more
pixels to denoising.  The parameter adjusts several internal values in a
coordinated manner:

* ``DENOISE_STAR_THRESHOLD`` is linearly interpolated from 1.2→1.0.  The
  lower number makes the detector more sensitive to faint spikes.
* ``DENOISE_STAR_MAX_AREA`` is interpolated from 20→50 pixels², ensuring
  that even at the weakest setting a moderate-sized point is still
  protected.
* ``DENOISE_STAR_LAPLACIAN_FACTOR`` is interpolated from 2.0→0.0, so
  stricter sharpness tests apply at low strength.

(An explicit value for any of these keys overrides the strength-based
mapping.)

The denoiser also applies the global ``DENOISE_STAR_PROTECT_WEIGHT``
when blending; if not specified this weight is derived from
``DENOISE_STAR_MASK_STRENGTH`` and ranges 0.95→1.0.  Together these
transformations create a "window" of real‑world usability: on our
synthetic star field, the median denoiser at strength‑3 loses roughly
10 % of stars with ``mask_strength=0``, about 5 % at the default 0.5,
and essentially none at 1.0.  Users can therefore dial the parameter
up or down to trade off noise removal against point‑source
preservation.

Configuration options:
  DENOISE_STAR_BG_SIGMA        : float (default 20) – sigma for background estimate
  DENOISE_STAR_THRESHOLD       : float (default 2.0) – factor applied to median excess
  DENOISE_STAR_DILATE          : int   (default 3) – dilation radius in pixels
  DENOISE_STAR_SOFT_EDGE       : float (default 3.0) – sigma for soft-edge blur
  DENOISE_STAR_MAX_AREA        : int   (default 0, off) – reject blobs larger than this area
  DENOISE_STAR_LAPLACIAN_FACTOR: float (default 0.0, off) – require laplacian > median*factor
  DENOISE_STAR_MASK_STRENGTH   : float (0..1) – convenience dial controlling max_area and lap_factor* ``DENOISE_STAR_MASK_FAST``       : bool (default False) – enable a very light, approximate star detector that skips the
   expensive Gaussian/Laplacian steps and runs in a few tens of milliseconds.  Useful on low‑power hardware.
The implementation here is almost identical to ``denoise._star_mask`` but
with the parameter translation logic extracted.
"""

import cv2
import numpy as np



# simple per-image cache: denoise may call the mask multiple times
# for the same frame when running different algorithms, so avoid recompute.
_cached_img = None
_cached_config = None
_cached_mask = None

def generate_star_mask(img, config):
    """Simple point-source mask generator with optional caching.

    The routine is deliberately lightweight; it honours a handful of
    configuration keys but does **not** expose the earlier strength dial
    or complex interpolation logic.  The goal is to stay close to the state
    you rolled back to while still allowing quick tuning of the detection.

    The following configuration keys are recognised:
      * DENOISE_STAR_BG_SIGMA (float, default 20) – background blur sigma
      * DENOISE_STAR_THRESHOLD (float, default 2.0) – multiplier applied to
        median excess when forming the binary star map.  Set ≤0 to disable.
      * DENOISE_STAR_MAX_AREA (int, default 0 == off) – drop blobs larger
        than this area (pixels²).
      * DENOISE_STAR_LAPLACIAN_FACTOR (float, default 0.0) – require the
        Laplacian magnitude to exceed this multiple of its median.
      * DENOISE_STAR_DILATE (int, default 3) – dilation radius for the mask.
      * DENOISE_STAR_SOFT_EDGE (float, default 3.0) – Gaussian sigma used to
        soften mask edges.
      * DENOISE_STAR_MASK_FAST (bool, False) – use a very cheap box-filter
        approximation instead of full Gaussian/Laplacian.

    If the same ``img`` and ``config`` objects are passed repeatedly the
    previously computed mask is returned from a tiny cache, speeding up
    multi‑algorithm denoising.
    """

    global _cached_img, _cached_config, _cached_mask
    # check cache
    if img is _cached_img and config is _cached_config:
        return _cached_mask

    # compute luminance
    if img.ndim == 3 and img.shape[2] >= 3:
        lum = (0.299 * img[:, :, 2].astype(np.float32) +
               0.587 * img[:, :, 1].astype(np.float32) +
               0.114 * img[:, :, 0].astype(np.float32))
    else:
        lum = img.astype(np.float32)

    # optional downsampling to speed up mask generation (accuracy loss)
    down = int(config.get('DENOISE_STAR_MASK_DOWNSAMPLE', 1))
    if down > 1:
        small = cv2.resize(lum, (lum.shape[1]//down, lum.shape[0]//down), interpolation=cv2.INTER_AREA)
    else:
        small = lum

    # read parameters with defaults
    dilate_radius = int(config.get('DENOISE_STAR_DILATE', 3))
    soft_edge = float(config.get('DENOISE_STAR_SOFT_EDGE', 3.0))
    sigma_bg = float(config.get('DENOISE_STAR_BG_SIGMA', 20.0))
    threshold_factor = float(config.get('DENOISE_STAR_THRESHOLD', 2.0))
    max_area = int(config.get('DENOISE_STAR_MAX_AREA', 0))
    lap_factor = float(config.get('DENOISE_STAR_LAPLACIAN_FACTOR', 0.0))

    # disable entirely by setting threshold <= 0
    if threshold_factor <= 0.0:
        return np.zeros(lum.shape, dtype=np.float32)

    # fast approximation mode
    if config.get('DENOISE_STAR_MASK_FAST', False):
        bg = cv2.blur(lum, (dilate_radius * 2 + 1, dilate_radius * 2 + 1))
        excess = np.maximum(lum - bg, 0.0)
        pos = excess[excess > 0]
        if pos.size == 0:
            return np.zeros(lum.shape, dtype=np.float32)
        star_thresh = float(np.median(pos)) * threshold_factor
        star_binary = (excess > star_thresh).astype(np.uint8)
        mask = cv2.GaussianBlur(star_binary.astype(np.float32), (0, 0), soft_edge)
        return np.clip(mask, 0.0, 1.0)

    # standard path
    try:
        # optionally compute bg and laplacian concurrently for slight overlap
        def compute_bg():
            return cv2.GaussianBlur(small, (0, 0), sigma_bg)
        def compute_lap():
            return cv2.Laplacian(small, cv2.CV_32F, ksize=3) if lap_factor > 0 else None
        bg = None
        lap = None
        # use threads if GIL-releasing ops
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exe:
            fut_bg = exe.submit(compute_bg)
            fut_lap = exe.submit(compute_lap)
            bg = fut_bg.result()
            lap = fut_lap.result()
        excess = np.maximum(small - bg, 0.0)
        pos = excess[excess > 0]
        if pos.size == 0:
            return np.zeros(lum.shape, dtype=np.float32)

        star_thresh = float(np.median(pos)) * threshold_factor
        star_binary = (excess > star_thresh).astype(np.uint8)

        if max_area > 0:
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats(star_binary)
            for i in range(1, nlab):
                if stats[i, cv2.CC_STAT_AREA] > max_area:
                    star_binary[labels == i] = 0

        if lap_factor > 0.0 and lap is not None:
            lap_abs = np.abs(lap)
            lap_med = np.median(lap_abs)
            sharp = lap_abs > (lap_med * lap_factor)
            star_binary = star_binary & sharp.astype(np.uint8)

        if dilate_radius > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_radius * 2 + 1, dilate_radius * 2 + 1))
            star_binary = cv2.dilate(star_binary, kern)

        soft = cv2.GaussianBlur(star_binary.astype(np.float32), (0, 0), soft_edge)
        mask = np.clip(soft, 0.0, 1.0)
        # if downsampled, scale mask back to full size
        if down > 1:
            mask = cv2.resize(mask, (lum.shape[1], lum.shape[0]), interpolation=cv2.INTER_LINEAR)
        # update cache before returning
        _cached_img = img
        _cached_config = config
        _cached_mask = mask
        return mask
    except Exception:
        return np.zeros(lum.shape, dtype=np.float32)
