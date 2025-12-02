from scale import get_normalized_hand_landmarks
import numpy as np

A = get_normalized_hand_landmarks("test_images/dumb_hand_sign.jpg")
B = get_normalized_hand_landmarks("test_images/IMG_5375.jpg")
D = get_normalized_hand_landmarks("test_images/hudson_ino_jutsu.jpg")

