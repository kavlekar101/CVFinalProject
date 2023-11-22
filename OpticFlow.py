import numpy as np
import scipy

def calculate_forward_optic_flow(Im, Im2):
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Ix = scipy.ndimage.convolve(Im, Gx, mode='nearest')
    Iy = scipy.ndimage.convolve(Im, Gy, mode='nearest')
    
    It = Im2 - Im
    
    Ix_flatten = Ix.flatten()
    Iy_flatten = Iy.flatten()
    
    A = np.column_stack((Ix_flatten, Iy_flatten))
    b = It.flatten()
    
    res = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
    u, v = res[0], res[1]
    return u, v
    
    