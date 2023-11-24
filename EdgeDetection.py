import numpy as np
import math
import skimage
import scipy
import cv2

def gaussDeriv2D(sigma):
    size = math.ceil(3*sigma) * 2 + 1
    half = size // 2
    image = np.zeros((size,size))
    Gx = np.zeros((size,size))
    Gy = np.zeros((size,size))
    
    for index, row in enumerate(image):
        for i in range(len(row)):
            Gx[index][i] = (((i - half)) / (2*math.pi*sigma**4)) * math.exp(-((i - half)**2 + (index - half)**2)/(2*sigma**2))
            Gy[index][i] = (((index - half)) / (2*math.pi*sigma**4)) * math.exp(-((i - half)**2 + (index - half)**2)/(2*sigma**2))
    
    return Gx, Gy

def calculateGradientsAndFindEdges(Im):
    Gx, Gy = gaussDeriv2D(2)
    
    greyIm = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    
    # you have to normalize it or else the image will be all black
    greyIm = cv2.normalize(greyIm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    gxIm = scipy.ndimage.convolve(greyIm, Gx, mode='nearest')
    gyIm = scipy.ndimage.convolve(greyIm, Gy, mode='nearest')
    
    
    magIm = np.sqrt(gxIm**2 + gyIm**2)