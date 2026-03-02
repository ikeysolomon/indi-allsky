import cv2
import numpy as np
from indi_allsky.protection_masks import star_mask, _cached_star

img = cv2.imread(r'DENOISE PR TEST ENVIRONMENT\ccd1_20260301_224251.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

def count_stars(mask):
    if mask.min() < 0.5:
        binary = (mask < 0.5).astype('uint8')
        n,_ = cv2.connectedComponents(binary)
        return n-1
    return 0

cases = [
    dict(name='default test', kwargs={'min_peak_ratio':0.12, 'min_sharpness':0.25,'max_sharpness':5.0,'max_elongation':2.5}),
    dict(name='no_psf_filters', kwargs={'min_peak_ratio':None, 'min_sharpness':None,'max_sharpness':None,'max_elongation':None}),
    dict(name='low_thresh', kwargs={'threshold_sigma':1.0, 'min_peak_ratio':0.12}),
    dict(name='low_thresh_nofilters', kwargs={'threshold_sigma':1.0}),
    dict(name='high_thresh', kwargs={'threshold_sigma':3.0}),
]

for c in cases:
    m = star_mask(img, percentile=99.0, **c['kwargs'])
    print(c['name'], 'stars=', count_stars(m), 'min,max,unique=', m.min(), m.max(), len(np.unique(m)))
