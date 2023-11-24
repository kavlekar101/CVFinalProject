import cv2
import time
import numpy as np
from skimage import io
import skimage
import scipy
import MeanShiftTracking as ms
import OpticFlow as of
import EdgeDetection as ed

start_time = time.time()
duration = 5  # Duration in seconds

# Path to your MP4 file
video_path = './redballbowled.mp4'

# Path to your png file
image_path = './ball.png'
im = io.imread(image_path)

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('frame')
cv2.startWindowThread()

# Check if video opened successfully
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
    # Check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates of point clicked: x = {x}, y = {y}")
        # You can also add code here to mark the clicked point on the frame
        # and display it again if needed.

# find the component of a along b
def find_component(a, b):
    # Define the vectors
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    b /= np.linalg.norm(b)

    # Calculate the dot product of a and b
    dot_product = np.dot(a, b)

    # Calculate the component of a along b
    component_along_b = dot_product * b

    # print("Component of a along b:", component_along_b)
    
    return component_along_b

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if i == 0:
        # # Create a window
        # cv2.namedWindow("frame")

        # # Set the mouse callback function for the window
        # cv2.setMouseCallback("frame", click_event)

        # # Display the first frame
        # cv2.imshow("frame", frame)

        # # Wait indefinitely until a key is pressed
        # cv2.waitKey(0)
        # break
        
        # x, y = 555, 315
        # x, y = 1029, 477
        x, y = 896, 274
        # y, x = ct.find_ball(im, frame)
        half_width = 20
        half_height = 20
        
        edge_frame = ed.calculateGradientsAndFindEdges(frame)
        # cv2.imwrite('initial_frame.jpg', frame)
        
        cv2.rectangle(frame, (x - half_width, y - half_height), (x + half_width, y + half_height), (0, 0, 0), 2)
        # cv2.imwrite('path_to_save_image.jpg', frame)
        frames.append(frame)
        i += 1
    else:
        # do the mean shift tracking
        edge_frame = ed.calculateGradientsAndFindEdges(frame)
        new_x, new_y = ms.meanshiftTracking(frames[i-1], edge_frame, x, y)
        x_diff = int(new_x - x) + 20
        y_diff = int(new_y - y) + 20
        
        frames.append(frame)
        i += 1
        
        img1 = frames[i-2][int(y):int(y)+y_diff, int(x):int(x)+x_diff]
        img2 = frames[i-1][int(y):int(y)+y_diff, int(x):int(x)+x_diff]
        
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
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the video capture object
cap.release()



cv2.waitKey(1)
# Closes all the frames
cv2.destroyAllWindows()
cv2.waitKey(1)