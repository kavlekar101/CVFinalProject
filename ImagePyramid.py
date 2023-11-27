import scipy
import numpy as np

def create_pyramid(frame, levels=4):
    pyramid = [frame]
    for i in range(levels):
        pyramid.append(downsample(pyramid[i], 2))
    return pyramid

def downsample(frame, factor, sigmaL=1):
    scipy.ndimage.gaussian_filter(frame, sigmaL, mode='nearest', truncate=3)
    return frame[::factor, ::factor]