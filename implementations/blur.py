import numpy as np

def median_blur(image):
    h,w,_ = image.shape
    
    k = 5
    r = k//2
    blured = image.copy()
    for x in range(k, h-k):
        for y in range(k, w-k):
            blured[x,y] = np.median(image[x-r:x+r, y-r:y+r], axis=(0,1))
            
    return blured


def gaussian_blur(image):
    h,w,_ = image.shape
    
    k = 5
    r = k//2
    G = gaussian_kernel(k)
    G3 = np.array([G, G ,G]).reshape((5,5,3))
    
    blured = image.copy()
    for x in range(k, h-k):
        for y in range(k, w-k):
            blured[x,y] = np.sum(image[x-r-1:x+r, y-r-1:y+r] * G3, axis=(0,1)) 
            
            
    return blured
    

def gaussian_kernel(kernel_size: int, sigma=1):
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)