from indi_allsky.protection_masks import nebula_mask
import cv2, os, numpy as np
p = os.path.join('DENOISE PR TEST ENVIRONMENT','ccd1_20260301_224251.png')
img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
if img is None:
    print('image not found:', p)
    raise SystemExit(1)
m = nebula_mask(img.astype('float64'))
coverage = (m > 0.5).mean() * 100
print('nebula mask coverage % (mask>0.5):', coverage)
print('mask min,max,unique:', float(m.min()), float(m.max()), len(np.unique(m)))
