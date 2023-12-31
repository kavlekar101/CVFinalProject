import cv2
import time
import numpy as np
from skimage import io
import skimage
import scipy
import MeanShiftTracking as ms
import OpticFlow as of
import EdgeDetection as ed
import NCC as ncc
import CovarianceTracking as ct

start_time = time.time()
duration = 5  # Duration in seconds

video_path = './redballbowled.mp4'

cap = cv2.VideoCapture(video_path)
cv2.namedWindow('frame')
cv2.startWindowThread()

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frames = []
i = 0
y, x = 0, 0

# x, y = 834, 696 # topright part of cricket field
# x, y = 1019, 754 # bottmright part of cricket field

# x, y = 807, 690
# x, y = 1830, 1020
vector_from_stump_to_stump = np.array([1830 - 807, 1020 - 690])
# the length along a cricket field is 22 yards

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates of point clicked: x = {x}, y = {y}")

# find the component of a along b
def find_component(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    b /= np.linalg.norm(b)

    dot_product = np.dot(a, b)

    # Calculate the component of a along b
    component_along_b = dot_product * b

    
    return component_along_b

while True:
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if i == 0:
        
        # x, y = 1029, 477
        x, y = 896, 274
        # x, y = 295, 106
        half_width = 20
        half_height = 20
        
        edge_frame = ed.calculateGradientsAndFindEdges(frame)
        cv2.imwrite('redballBetterOne.jpg', frame[y - half_height:y + half_height, x - half_width:x + half_width])
        
        cv2.rectangle(frame, (x - half_width, y - half_height), (x + half_width, y + half_height), (0, 0, 0), 2)
        # cv2.imwrite('path_to_save_image.jpg', frame)
        frames.append(frame)
        i += 1
    else:
        image_path = './redballBetterOne.jpg'
        im = io.imread(image_path)
        # do the mean shift tracking
        edge_frame = ed.calculateGradientsAndFindEdges(frame)
        
        # try this with better tracking so that we iterativly update the window
        new_x, new_y = ms.meanshiftTracking(frames[i-1], frame, x, y)
        print("done with meanshift")
        new_y_1, new_x_1 = ncc.pyramid_calculations(im, frame)
        print("done with ncc")
        new_y_2, new_x_2 = ct.find_ball(im, frame)
        print("done with covariance tracking")
        
        print(f"Distance Between Meanshift and NCC: {np.linalg.norm(np.array([new_x, new_y]) - np.array([new_x_1, new_y_1]))}")
        print(f"Distance Between Meanshift and Covariance Tracking: {np.linalg.norm(np.array([new_x, new_y]) - np.array([new_x_2, new_y_2]))}")
        
        x_diff = abs(int(new_x - x)) + 20
        y_diff = abs(int(new_y - y)) + 20
        
        frames.append(frame)
        i += 1
        
        y_frame = min(int(y), int(new_y))
        x_frame = min(int(x), int(new_x))
        img1 = frames[i-2][y_frame:y_frame+y_diff, x_frame:x_frame+x_diff]
        img2 = frames[i-1][y_frame:y_frame+y_diff, x_frame:x_frame+x_diff]
        
        x, y = new_x, new_y
        
        half_width = 20
        half_height = 20
        cv2.rectangle(frame, (int(x) - half_width, int(y) - half_height), (int(x) + half_width, int(y) + half_height), (0, 0, 0), 2)
        # I also have to define a window around the ball so that I can find the optic flow arbitrarily I am going to set it
        u, v = of.calculate_forward_optic_flow(img1, img2) # the first one is u and the second one is v
        optic_flow = np.array([u, v])
        
        component = find_component(optic_flow, vector_from_stump_to_stump)

        component_mag = np.linalg.norm(component)
        field_mag = np.linalg.norm(vector_from_stump_to_stump)
        
        speed = component_mag * 22 / field_mag
        print(f"Speed of the ball: {speed * fps} yards per second")
        
        component_without_optic_flow = find_component(vector_from_stump_to_stump, np.array([x_diff, y_diff]))
        component_without_optic_flow_mag = np.linalg.norm(component_without_optic_flow)
        speed_without_optic_flow = component_without_optic_flow_mag * 22 / field_mag
        print(f"Speed of the ball without optic flow: {speed_without_optic_flow * fps} yards per second")
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()



cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)