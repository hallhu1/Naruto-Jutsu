from landmark_identifier import get_normalized_hand_landmarks
import numpy as np

image_name = "chidori"

image_filepath = "test_images/" + image_name + ".jpg"
A = get_normalized_hand_landmarks(image_filepath)
jutsu_name = "groundtruths/" + image_name + ".npy"
np.save(jutsu_name, A)