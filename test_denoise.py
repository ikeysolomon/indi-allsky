from indi_allsky.denoise import IndiAllskyDenoise
import numpy as np

cfg = {}
d = IndiAllskyDenoise(cfg, [False])
mask = d._star_mask(np.zeros((10,10), dtype=np.uint8))
print('mask shape', mask.shape)
