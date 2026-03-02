import cv2, math
import numpy as np
from indi_allsky.protection_masks import _fast_bkg_stats, _cv2_find_stars

img = cv2.imread(r'DENOISE PR TEST ENVIRONMENT\ccd1_20260301_224251.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
med, std = _fast_bkg_stats(img, sigma=3.0)
print('med,std', med, std)
xs, ys = _cv2_find_stars(img, 4.5, 2.0, med, std, 0.12, 0.25, 5.0, 2.5)
print('n peaks', len(xs))

h,w = img.shape
mask = np.zeros_like(img, dtype=np.float32)
if len(xs)>0:
    stddev = 4.5 / 2.3548
    radius = int(math.ceil(4.5 * 3))
    ksize = 2*radius+1
    impulse = np.zeros_like(img, dtype=np.float32)
    xi = xs.astype(np.intp); yi = ys.astype(np.intp)
    valid = (xi>=0)&(xi<w)&(yi>=0)&(yi<h)
    impulse[yi[valid], xi[valid]] = 1.0
    blurred = cv2.GaussianBlur(impulse, (ksize, ksize), stddev)
    k1d = cv2.getGaussianKernel(ksize, stddev)
    peak = float(k1d[radius][0] ** 2)
    print('radius,ksize,peak', radius, ksize, peak)
    if peak>0:
        blurred *= (1.0/peak)
    print('blurred max', blurred.max())
    np.minimum(blurred, 1.0, out=blurred)
    np.maximum(mask, blurred, out=mask)
print('mask max after', mask.max(), 'unique', len(np.unique(mask)))
print('returned mask min', 1.0 - mask.max())
