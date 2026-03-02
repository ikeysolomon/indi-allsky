import cv2
import numpy as np
from indi_allsky.protection_masks import _fast_bkg_stats

img = cv2.imread(r'DENOISE PR TEST ENVIRONMENT\ccd1_20260301_224251.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
for i in range(5):
    med,std = _fast_bkg_stats(img, sigma=3.0)
    print(i, med, std)
