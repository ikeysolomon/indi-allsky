import cv2, numpy as np
from indi_allsky.protection_masks import star_mask
img=cv2.imread(r'DENOISE PR TEST ENVIRONMENT/ccd1_20260301_224251.png', cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')
s=star_mask(gray, percentile=99.0)
print('min',float(s.min()),'max',float(s.max()),'unique',len(np.unique(s)),'protected',int((s<0.5).sum()),'shape',s.shape)
