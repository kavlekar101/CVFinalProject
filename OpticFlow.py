import numpy as np
import scipy
import cv2

def calculate_forward_optic_flow(Im, Im2):
    GreyIm = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    GreyIm2 = cv2.cvtColor(Im2, cv2.COLOR_BGR2GRAY)
    
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Ix = scipy.ndimage.convolve(GreyIm, Gx, mode='nearest')
    Iy = scipy.ndimage.convolve(GreyIm, Gy, mode='nearest')
    
    It = GreyIm2 - GreyIm
    
    Ix_flatten = Ix.flatten()
    Iy_flatten = Iy.flatten()
    
    A = np.column_stack((Ix_flatten, Iy_flatten))
    b = It.flatten()
    
    res, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    u, v = res[0], res[1]
    return u, v
    
    