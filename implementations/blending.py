import numpy as np

def blend_image(im1: np.ndarray, im2: np.ndarray, alpha, beta, gamma):
    result = alpha * im1.astype(np.float32) + beta * im2.astype(np.float32) + gamma
    return np.clip(result, 0, 255).astype(np.uint8)