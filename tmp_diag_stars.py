import cv2
import numpy as np
from indi_allsky.protection_masks import _fast_bkg_stats, _cv2_find_stars

img = cv2.imread(r'DENOISE PR TEST ENVIRONMENT\ccd1_20260301_224251.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
print('shape', img.shape)
med, std = _fast_bkg_stats(img, sigma=3.0)
print('bkg median, std:', med, std)
xs, ys = _cv2_find_stars(img, fwhm=4.5, threshold_sigma=2.0, bkg_median=med, bkg_std=std)
print('found', len(xs))
# Print some stats of smoothed image inside _cv2_find_stars logic by recomputing
import math
fwhm=4.5
stddev = fwhm / 2.3548
radius = int(math.ceil(fwhm * 2.5))
ksize = 2*radius+1
sm = cv2.GaussianBlur(img, (ksize, ksize), stddev)
print('smoothed max, mean, p99:', float(sm.max()), float(sm.mean()), float(np.percentile(sm,99)))

# Show abs_thresh
abs_thresh = med + 2.0*std
print('abs_thresh', abs_thresh)

# show number of pixels above threshold
print('pixels > abs_thresh:', np.count_nonzero(sm > abs_thresh))
print('nonzero pixels in sm>dilated?', np.count_nonzero(sm> (cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))))) )
