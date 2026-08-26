import numpy as np

def binary_thresholding_inv(image, threshold):
    return 255 - binary_thresholding(image, threshold)

def binary_thresholding(image, threshold):
    binary_image = (image > threshold).astype(np.uint8) * 255
    return binary_image

def trunc_thresholding(image, threshold):
    trunc_mask = image > threshold
    result = image.copy()    
    
    result[trunc_mask] = threshold
    return result

def to_zero_thresholding_inv(image, threshold):
    to_zero_mask = image > threshold
    result = image.copy()    
    
    result[to_zero_mask] = 0
    return result

def to_zero_thresholding(image, threshold):
    to_zero_mask = image > threshold
    result = image.copy()    
    
    result[~to_zero_mask] = 0
    return result


def gaussian_thresholding(image: np.ndarray, C: float, kernel_size: int = 5) -> np.ndarray:
    image = image.astype(np.float32)
    
    kernel = gaussian_kernel(kernel_size=kernel_size, sigma=3)
    
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    wa = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            wa[i, j] = np.sum(window * kernel)
            
    T = wa - C

    result = image.copy()
    mask = result > T
    result[mask] = np.max(image)
    result[~mask] = 0
    return result

def gaussian_kernel(kernel_size: int, sigma=1):
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)