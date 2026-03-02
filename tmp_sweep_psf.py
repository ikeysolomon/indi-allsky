import cv2
import numpy as np
from indi_allsky.protection_masks import star_mask
img = cv2.imread(r'DENOISE PR TEST ENVIRONMENT\ccd1_20260301_224251.png', cv2.IMREAD_GRAYSCALE).astype('float32')

peak_vals = [None, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
sharp_vals = [None, 0.02, 0.05, 0.1, 0.2]
elong_vals = [None, 2.5, 4.0, 6.0]

results = []
for pv in peak_vals:
    for sv in sharp_vals:
        for ev in elong_vals:
            kwargs = {}
            if pv is not None:
                kwargs['min_peak_ratio'] = pv
            if sv is not None:
                kwargs['min_sharpness'] = sv
            if ev is not None:
                kwargs['max_elongation'] = ev
            m = star_mask(img, percentile=99.0, **kwargs)
            if m.min()<0.5:
                binary=(m<0.5).astype('uint8')
                n,_=cv2.connectedComponents(binary)
                stars = n-1
            else:
                stars = 0
            pct = np.count_nonzero(m<0.5)/m.size*100
            results.append((pv,sv,ev,stars,pct))

# print combos with stars>0 sorted
for row in sorted(results, key=lambda r: (-r[3], r[4]))[:40]:
    print(row)

# find first combo with >100 stars and pct<10
for row in results:
    pv,sv,ev,stars,pct = row
    if stars>100 and pct<10:
        print('\nGood candidate:', row)
        break
