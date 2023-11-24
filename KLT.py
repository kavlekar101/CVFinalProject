import skimage      
from skimage import io, filters
from skimage import filters
from scipy.ndimage import sobel, interpolation
import skimage.color as color
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import eig
import cv2
import time

def read_video(file_path):
    cap = cv2.VideoCapture(file_path)
    # start_time = time.time()
    # duration = 5
    frames = []

    # Set the start position to 200 milliseconds
    start_time_milliseconds = 200
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_milliseconds)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Convert to grayscale
    cap.release()
    
    return frames


frame = io.imread('frame.jpg')
gray_frame = color.rgb2gray(frame)


sigmaI = 1
sigmaD = 0.7

def gaussDeriv2D(sigma):

    min = -np.ceil(3 * sigma)
    max = np.ceil(3 * sigma)

    xRange = np.arange(min, max + 1)    
    yRange = np.arange(min, max +1)

    X = np.array([[x for x in xRange] for y in yRange])
    Y =  np.array([[x for x in xRange] for y in yRange]).T

    Gx = ((X - 0) / (2 * np.pi * sigma**4)) * np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * sigma**2))
    Gy = ((Y - 0) / (2 * np.pi * sigma**4)) * np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * sigma**2))
    return Gx, Gy


Gx, Gy = gaussDeriv2D(sigmaD)

Ix = scipy.ndimage.convolve(gray_frame, Gx, mode='nearest')
Iy = scipy.ndimage.convolve(gray_frame, Gy, mode='nearest')

IxIx = Ix * Ix
IyIy = Iy * Iy
IxIy = Ix * Iy

Ix2 = scipy.ndimage.gaussian_filter(IxIx, sigmaI, mode='nearest', truncate=3)
Iy2 = scipy.ndimage.gaussian_filter(IyIy, sigmaI, mode='nearest', truncate=3)
IxIy = scipy.ndimage.gaussian_filter(IxIy, sigmaI, mode='nearest', truncate=3)

a = 0.05

RImg = (Ix2 * Iy2 - IxIy**2) - a * (Ix2 + Iy2)**2
# print(RImg[250, 100])
# plt.title('RImg')
# plt.imshow(RImg, cmap='gray')
# plt.show()

rValues = RImg[15:22, 15:22] 
# print(rValues)
# plt.title('R values')
# plt.imshow(rValues, cmap='gray')
# plt.show()

threshold = -4.96168452402024e-07
RImg[RImg >= threshold] = 0

new = RImg[100:150, 350:400]
# print(new)
# plt.imshow(new, cmap='gray')
# plt.title('new')
# plt.show()
# print(new[12,20])

# plt.imshow(RImg, cmap='gray')
# plt.title('R')
# plt.show()

ballFeaturePoints = []

h = RImg.shape[0]
w = RImg.shape[1]
points = []

new_matrix = np.zeros((h, w))

for y in range(h - 3 + 1):
    for x in range(w - 3 + 1):
        region = RImg[y:y + 3, x:x + 3]
        maxPoint = np.max(region)
        count = 0

        for i in region.flatten():
            if i == maxPoint:
                count += 1

        if count == 1:
            # Update the new matrix at the position (y, x) with the value from RImg
            new_matrix[y, x] = RImg[y, x]
            # Add the point to the list
            points.append((y, x))


# plt.imshow(new_matrix, cmap='gray')
# plt.title('new_matrix')
# plt.show()

threshold2 = -4.96168452402024e-07

new_matrix[new_matrix >= threshold2] = 0
close_up= new_matrix[100:150, 350:400]
# plt.imshow(close_up, cmap='gray')
# plt.title('close_up')
# plt.show()
# print(close_up[12,20])

for y in range(RImg.shape[0]):
    for x in range(RImg.shape[1]):
        if new_matrix[y, x] != 0:
            if (y >= 100 and y < 150) and (x >= 350 and x < 400):
                ballFeaturePoints.append((y, x))

# plt.imshow(frame, cmap='gray')
# plt.plot([y[1] for y in ballFeaturePoints], [x[0] for x in ballFeaturePoints], 'r.', markersize=0.1)  
# plt.title('double check')
# plt.show()

# print(len(ballFeaturePoints))

def compute_gradients(frame):
    Ix = filters.sobel_h(frame)
    Iy = filters.sobel_v(frame)
    return Iy, Ix


def track_point(y, x, Iy1, Ix1, It, window_size):
    A = []
    b = []
    half_window = window_size // 2  # Calculate half the window size
    print(Iy1.shape)
    print("Point:", x, y)


    # print("x", x)
    # print("y", y)
    # print("Ix1", Ix1)
    # print("Iy1", Iy1)
    # print("It", It)
    
    for dy in range(-half_window, half_window + 1):
        for dx in range(-half_window, half_window + 1):
            px, py = x + dx, y + dy
            print("px", px)
            print("py", py)
        
            if 0 <= py < Iy1.shape[0] and 0 <= px < Ix1.shape[1]:
                print("here")
                A.append([Iy1[py, px], Ix1[py, px]])
                b.append(-It[py, px])
    
    A = np.array(A)
    # print(A)
    # # print(A.shape, "ashape")

    b = np.array(b)
    
    # print(b.shape, "bshape")


    # if A.shape[0] < 2 or A.shape[0] != b.shape[0]:
    # # Not enough data or mismatched dimensions, return no movement
    #     return (0, 0)
    
    nu = np.linalg.lstsq(A, b, rcond=None)[0]  # nu = [u, v]
    return nu



def track_points_in_frame(frame1, frame2, points_to_track, window_size):
    Iy1, Ix1 = compute_gradients(frame1)
    # plt.imshow(Ix1, cmap='gray')
    # plt.title('Ix1')
    # plt.show()

    It = frame2 - frame1
    # plt.imshow(It, cmap='gray')
    # plt.title('It')
    # plt.show()

    displacements = [track_point(y, x, Iy1, Ix1, It, window_size) for y, x in points_to_track]
    updated_points = [(y + int(v), x + int(u)) for (y, x), (u, v) in zip(points_to_track, displacements)]
    return updated_points

def visualize_tracking(frame, points, frame_number):
    for y, x in points:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.imshow('Tracking', frame)
    window_title = "Frame Number: " + str(frame_number)
    cv2.setWindowTitle('Tracking', window_title)



frames = read_video('redBall.mp4')
windowSize = 5
first_frame = frames[0]

# print(len(ballFeaturePoints))
frames = [first_frame, frames[1]]



for i in range(len(frames) - 1):
    # print(twoFrames[i].shape, twoFrames[i + 1].shape)
    print("Frame:", i)
    
    visualize_tracking(frames[i], ballFeaturePoints, i)
    updated_points = track_points_in_frame(frames[i], frames[i + 1], ballFeaturePoints, windowSize)
    ballFeaturePoints = updated_points

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.waitKey(1)
cv2.destroyAllWindows() 







