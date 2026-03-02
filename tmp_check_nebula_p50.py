from indi_allsky.protection_masks import nebula_mask, star_mask
import cv2, os, numpy as np
out_dir = os.path.join('DENOISE PR TEST ENVIRONMENT', 'Test Output')
img_p = os.path.join('DENOISE PR TEST ENVIRONMENT','ccd1_20260301_224251.png')
img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
if img is None:
    print('image not found:', img_p)
    raise SystemExit(1)
# compute star mask first
s = star_mask(img.astype(np.float32), percentile=99.0)
# tuned nebula mask
m = nebula_mask(img.astype('float64'), star_m=s, percentile=50.0)
# stats
coverage = (m > 0.5).mean() * 100
print('nebula mask coverage % (mask>0.5):', coverage)
print('mask min,max,unique:', float(m.min()), float(m.max()), len(np.unique(m)))
# save visual
out_path = os.path.join(out_dir, 'real_03_nebula_mask_p50.png')
u8 = (np.clip(m,0,1)*255).astype('uint8')
cv2.imwrite(out_path, u8)
print('saved:', out_path)
