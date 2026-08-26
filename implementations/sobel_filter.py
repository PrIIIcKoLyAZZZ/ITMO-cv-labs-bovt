import numpy as np

def apply_sobel_op(image: np.ndarray):
    G_x = _apply_kernel(image, _get_horizontal_kernel())
    G_y = _apply_kernel(image, _get_vertical_kernel())
    
    return np.sqrt(G_x**2 + G_y**2)
    
    #return np.abs(G_x.astype(np.int8))+np.abs(G_y.astype(np.int8))
        
    
def _get_horizontal_kernel():
    return np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).astype(np.float32)

def _get_vertical_kernel():
    return np.array([[1,2,1],[0,0,0],[-1,-2,-1]]).astype(np.float32)


def _apply_kernel(image: np.ndarray, kernel: np.ndarray):
    h, w = image.shape
    
    result = np.zeros_like(image, shape=image.shape, dtype=np.float32)
    
    for x in range(1, w-1):
        for y in range(1, h-1):
            result[x,y] = np.sum(image[x-1:x+2, y-1:y+2] * kernel) 
            
    return result
            