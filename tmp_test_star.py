import numpy as np
from indi_allsky.protection_masks import star_mask

img = np.zeros((200,200), dtype=np.float32)
img[100,100] = 255.0
m = star_mask(img)
print(m.min(), m.max(), (m==0).sum())
