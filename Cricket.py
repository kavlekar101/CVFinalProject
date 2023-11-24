import cv2
import time
import numpy as np

start_time = time.time()
duration = 5  # Duration in seconds

# Path to your MP4 file
video_path = './CricketBallSpeedClip.mp4'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Check if the duration has been exceeded
    if time.time() - start_time > duration:
        break

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

# When everything done, release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
