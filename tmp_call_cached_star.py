import hashlib
import cv2
import numpy as np
from indi_allsky.protection_masks import _cached_star

img = cv2.imread(r'DENOISE PR TEST ENVIRONMENT\ccd1_20260301_224251.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
raw = img.tobytes()
key = hashlib.md5(raw).digest()
shape = img.shape
# args: key_hash, percentile, threshold_sigma=3.0, fwhm=4.5, min_peak_ratio=None, ... , shape=None, _raw_bytes
mask = _cached_star(key, 99.0, 2.0, 4.5, 0.12, 0.25, 5.0, 2.5, shape, _raw_bytes=raw)
print('mask min,max,unique=', mask.min(), mask.max(), len(np.unique(mask)))
