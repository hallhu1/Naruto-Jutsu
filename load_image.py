from landmark_identifier import get_normalized_hand_landmarks
import numpy as np
import cv2

image_name = "rasengan"

image_filepath = "test_images/" + image_name + ".jpg"
cv_image = cv2.imread(image_filepath)
A = get_normalized_hand_landmarks(cv_image)
jutsu_name = "groundtruths/" + image_name + ".npy"
np.save(jutsu_name, A)