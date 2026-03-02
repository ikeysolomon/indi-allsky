import cv2, numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from indi_allsky.protection_masks import star_mask

img=cv2.imread(r'DENOISE PR TEST ENVIRONMENT/ccd1_20260301_224251.png', cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')

# parameters used by current star_mask
threshold_sigma = 5.0
fwhm = 3.0

# background estimate
mean, median, std = sigma_clipped_stats(gray, sigma=3.0)
print('bkg mean, median, std=', mean, median, std)

thr = threshold_sigma * std
print('threshold (sigma*std)=', threshold_sigma, '*', std, '=', thr)

daofind = DAOStarFinder(fwhm=fwhm, threshold=thr)
tbl = daofind(gray)
print('DAOStarFinder detections:', 0 if tbl is None else len(tbl))
if tbl is not None:
    print('Sample detections (first 10):')
    for i,row in enumerate(tbl[:10]):
        print(i, float(row['xcentroid']), float(row['ycentroid']), float(row['peak']))

# star_mask via module
s = star_mask(gray, percentile=99.0)
print('star_mask min,max,unique,protected_pixels=', float(s.min()), float(s.max()), len(np.unique(s)), int((s<0.5).sum()))

# save overlay for quick inspection
overlay = img.copy()
if tbl is not None:
    for row in tbl:
        cv2.circle(overlay, (int(row['xcentroid']), int(row['ycentroid'])), int(fwhm), (0,255,0), 1)
cv2.imwrite(r'DENOISE PR TEST ENVIRONMENT/Test Output/tmp_dao_overlay.png', overlay)
cv2.imwrite(r'DENOISE PR TEST ENVIRONMENT/Test Output/tmp_star_mask.png', (np.clip(s*255,0,255)).astype('uint8'))
print('Wrote Test Output/tmp_dao_overlay.png and tmp_star_mask.png')
