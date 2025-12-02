from scale import get_normalized_hand_landmarks
import numpy as np
import cv2

# A = get_normalized_hand_landmarks("test_images/dumb_hand_sign.jpg")
# B = get_normalized_hand_landmarks("test_images/IMG_5377.jpg")
# D = get_normalized_hand_landmarks("test_images/hudson_ino_jutsu.jpg")
# E = get_normalized_hand_landmarks("test_images/alex_ino_jutsu.jpg")
# F = get_normalized_hand_landmarks("test_images/woman_hands.jpg")

# print(np.linalg.norm(A - B))
# print(np.linalg.norm(A - D))
# print(np.linalg.norm(A - E))
# print(np.linalg.norm(A - F))

# 0 is usually the default camera. Try 1, 2, ... if you have multiple cameras.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    
    # Show the frame
    cv2.imshow('Camera Feed', frame)


    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
