#!/usr/bin/env python3
"""Aggressive sweep of nebula_mask parameters to produce visible masks.
Saves outputs to DENOISE PR TEST ENVIRONMENT/Test Output/nebula_sweep_*.png
"""
from indi_allsky.protection_masks import nebula_mask, star_mask
import cv2, os, numpy as np, time

OUT = os.path.join('DENOISE PR TEST ENVIRONMENT','Test Output')
IMG = os.path.join('DENOISE PR TEST ENVIRONMENT','ccd1_20260301_224251.png')
img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
if img is None:
    print('image not found:', IMG)
    raise SystemExit(1)

s = star_mask(img.astype('float32'), percentile=99.0)

percentiles = [50, 40, 30, 20, 10]
box_sizes = [128, 64, 32]
filter_sizes = [(101,101), (51,51), (31,31), (11,11), (5,5)]

results = []
for p in percentiles:
    for b in box_sizes:
        for f in filter_sizes:
            start = time.perf_counter()
            try:
                m = nebula_mask(img.astype('float64'), star_m=s, percentile=p, box_size=b, filter_size=f)
            except Exception as e:
                print('error for', p, b, f, e)
                continue
            elapsed = (time.perf_counter()-start)
            cov = (m>0.5).mean()*100
            unique = len(np.unique(m))
            fname = f'nebula_p{p}_b{b}_f{f[0]}x{f[1]}.png'
            outp = os.path.join(OUT, fname)
            u8 = (np.clip(m,0,1)*255).astype('uint8')
            cv2.imwrite(outp, u8)
            results.append((cov, unique, p, b, f, fname, elapsed))
            print(f'p={p:02d} box={b:3d} filt={f[0]}x{f[1]} -> cov={cov:.2f}% uniq={unique} t={elapsed:.2f}s')

# show top results by coverage
results.sort(reverse=True)
print('\nTop results by coverage:')
for cov, uniq, p, b, f, fname, elapsed in results[:10]:
    print(f'{cov:6.2f}%  p={p} b={b} f={f[0]}x{f[1]} -> {fname} (uniq={uniq})')

print('\nSaved masks into', OUT)
