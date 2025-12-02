from scale import get_normalized_hand_landmarks
import numpy as np

image_name = "hudson_ino_jutsu"
jutsu_name = "ino"

image_name = "test_images/" + image_name + ".jpg"
A = get_normalized_hand_landmarks(image_name)
jutsu_name = "groundtruths/" + jutsu_name + ".npy"
np.save(jutsu_name, A)