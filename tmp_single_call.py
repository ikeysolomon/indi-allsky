import cv2
import numpy as np
from indi_allsky.protection_masks import star_mask

img = cv2.imread(r'DENOISE PR TEST ENVIRONMENT\ccd1_20260301_224251.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

m = star_mask(img, percentile=99.0, min_peak_ratio=0.06, min_sharpness=0.12, max_sharpness=10.0, max_elongation=4.0)
print('min,max,unique=', m.min(), m.max(), len(np.unique(m)))
if m.min() < 0.5:
    binary = (m<0.5).astype('uint8')
    n,_ = cv2.connectedComponents(binary)
    print('stars', n-1)
else:
    print('no stars')
