import os
import numpy as np
from landmark_identifier import get_normalized_hand_landmarks

def load_groundtruths(folder_path="groundtruths"):
    """
    Load all .npy files from groundtruths folder.
    
    Returns:
        dict mapping {filename_without_extension: numpy_array}
    """
    groundtruths = {}
    
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".npy"):
            name = os.path.splitext(fname)[0]   # strip ".npy"
            path = os.path.join(folder_path, fname)
            
            try:
                arr = np.load(path)
                groundtruths[name] = arr
            except Exception as e:
                print(f"Warning: failed to load {fname}: {e}")

    return groundtruths

def find_jutsu(groundtruths, input_img):
    query = get_normalized_hand_landmarks(input_img)

    distances = {}
    for name, gt_pose in groundtruths.items():
        distances[name] = np.linalg.norm(query - gt_pose)
    closest = min(distances, key=distances.get)
    return closest, distances[closest]

gt = load_groundtruths()
while True:
    test_image = input("Enter the image name: ")
    closest, distance = find_jutsu(gt, "test_images/" + test_image + ".jpg")
    print(closest)


