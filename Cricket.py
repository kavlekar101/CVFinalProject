import cv2
import time
import numpy as np
from skimage import io
import skimage
import CovarianceTracking as ct
import MeanShiftTracking as ms

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

def click_event(event, x, y, flags, param):
    # Check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates of point clicked: x = {x}, y = {y}")
        # You can also add code here to mark the clicked point on the frame
        # and display it again if needed.

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
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
        
        # do the covariance tracking
        # I have to reduce the frame size first
        # x, y = 555, 315
        x, y = 1029, 477
        # x, y = 896, 274
        # y, x = ct.find_ball(im, frame)
        half_width = 20
        half_height = 20
        cv2.rectangle(frame, (x - half_width, y - half_height), (x + half_width, y + half_height), (0, 0, 0), 2)
        # cv2.imwrite('path_to_save_image.jpg', frame)
    else:
        print(x, y)
        # do the mean shift tracking
        new_x, new_y = ms.meanshiftTracking(frames[i-1], frame, x, y)
        x, y = new_x, new_y
        
        half_width = 20
        half_height = 20
        cv2.rectangle(frame, (int(x) - half_width, int(y) - half_height), (int(x) + half_width, int(y) + half_height), (0, 0, 0), 2)
        # I also have to define a window around the ball so that I can find the optic flow arbitrarily I am going to set it to 40x40
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
    frames.append(frame)
    i += 1

# When everything done, release the video capture object
cap.release()

cv2.waitKey(1)
# Closes all the frames
cv2.destroyAllWindows()
cv2.waitKey(1)