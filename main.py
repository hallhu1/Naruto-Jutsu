from scale import get_normalized_hand_landmarks
import numpy as np

A = get_normalized_hand_landmarks("test_images/dumb_hand_sign.jpg")
B = get_normalized_hand_landmarks("test_images/IMG_5377.jpg")
D = get_normalized_hand_landmarks("test_images/hudson_ino_jutsu.jpg")
E = get_normalized_hand_landmarks("test_images/alex_ino_jutsu.jpg")
F = get_normalized_hand_landmarks("test_images/woman_hands.jpg")

print(np.linalg.norm(A - B))
print(np.linalg.norm(A - D))
print(np.linalg.norm(A - E))
print(np.linalg.norm(A - F))
