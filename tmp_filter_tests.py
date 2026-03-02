import cv2
import numpy as np
from indi_allsky.protection_masks import star_mask
img = cv2.imread(r'DENOISE PR TEST ENVIRONMENT\ccd1_20260301_224251.png', cv2.IMREAD_GRAYSCALE).astype('float32')
cases = [
    ('no_psf', {'min_peak_ratio':None,'min_sharpness':None,'max_sharpness':None,'max_elongation':None}),
    ('only_elong', {'min_peak_ratio':None,'min_sharpness':None,'max_sharpness':None,'max_elongation':4.0}),
    ('only_peak', {'min_peak_ratio':0.02,'min_sharpness':None,'max_sharpness':None,'max_elongation':None}),
    ('only_sharp', {'min_peak_ratio':None,'min_sharpness':0.05,'max_sharpness':None,'max_elongation':None}),
]
for desc, kwargs in cases:
    m = star_mask(img, percentile=99.0, **kwargs)
    if m.min()<0.5:
        binary=(m<0.5).astype('uint8')
        n,_=cv2.connectedComponents(binary)
        n=n-1
    else:
        n=0
    print(desc, 'stars=', n, 'min,max,unique=', m.min(), m.max(), len(np.unique(m)))
