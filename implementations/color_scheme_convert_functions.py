from matplotlib.pylab import Enum
import numpy as np


def RGB_to_HSV(image: np.array):
    M = np.max(image, axis=2)
    m = np.min(image, axis=2)
    C = M - m
        
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    
    V = M

    H = np.zeros_like(V, dtype=np.float32)

    mask_C = C != 0
    mask_r = mask_C & (M == r)
    mask_g = mask_C & (M == g)
    mask_b = mask_C & (M == b)

    H[mask_r] = ((g[mask_r] - b[mask_r]) / C[mask_r]) % 6
    H[mask_g] = ((b[mask_g] - r[mask_g]) / C[mask_g]) + 2
    H[mask_b] = ((r[mask_b] - g[mask_b]) / C[mask_b]) + 4
    H = (H * 60) % 360

    S_V = np.zeros_like(V, dtype=np.float32)
    valid = V != 0
    S_V[valid] = C[valid] / V[valid]

    return np.stack((H, S_V, V), axis=-1)
    

def RGB_to_HSL(image: np.array):
    M = np.max(image, axis=2)
    m = np.min(image, axis=2)
    C = M - m
    
    L = (M + m) / 2
        
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    
    V = M

    H = np.zeros_like(V, dtype=np.float32)

    mask_C = C != 0
    mask_r = mask_C & (M == r)
    mask_g = mask_C & (M == g)
    mask_b = mask_C & (M == b)

    H[mask_r] = ((g[mask_r] - b[mask_r]) / C[mask_r]) % 6
    H[mask_g] = ((b[mask_g] - r[mask_g]) / C[mask_g]) + 2
    H[mask_b] = ((r[mask_b] - g[mask_b]) / C[mask_b]) + 4
    H = (H * 60) % 360

    S_L = np.zeros_like(L, dtype=np.float32)
    denom = 1 - np.abs(2 * L - 1)
    valid = denom != 0
    S_L[valid] = C[valid] / denom[valid]

    return np.stack((H, S_L, L), axis=-1)