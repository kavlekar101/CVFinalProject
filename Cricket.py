import cv2
import time
import numpy as np

start_time = time.time()
duration = 5  # Duration in seconds

# Path to your MP4 file
video_path = './CricketBallSpeedClip.mp4'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('frame')
cv2.startWindowThread()

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the video capture object
cap.release()

cv2.waitKey(0)
# Closes all the frames
cv2.destroyAllWindows()
cv2.waitKey(1)