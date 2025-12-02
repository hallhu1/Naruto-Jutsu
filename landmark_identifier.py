import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import sys
import os
import contextlib


# ---- Create the detector ONCE, globally ----
def _create_two_hand_detector(model_path: str = "hand_landmarker.task"):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )
    return vision.HandLandmarker.create_from_options(options)

# Global singleton detector
HAND_DETECTOR = _create_two_hand_detector()

def get_normalized_hand_landmarks(image_path: str,
                                  scale_to: float = 1.0,
                                  detector: vision.HandLandmarker = HAND_DETECTOR
                                  ) -> np.ndarray:
    """
    Detect exactly two hands (Left and Right) and return a single (42, 3) array
    of normalized coordinates in a LEFT-hand-centric coordinate system.
    """
    # ---- 1) Run MediaPipe detector (up to 2 hands) ----
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    if len(hand_landmarks_list) != 2:
        raise ValueError(f"Expected exactly 2 hands, but detected {len(hand_landmarks_list)}.")

    # ---- 2) Identify Left and Right hands by handedness labels ----
    left_idx = None
    right_idx = None

    for i, handedness in enumerate(handedness_list):
        label = handedness[0].category_name  # "Left" or "Right"
        labell = label.lower()
        if labell.startswith("left"):
            left_idx = i
        elif labell.startswith("right"):
            right_idx = i

    if left_idx is None or right_idx is None:
        raise ValueError("Did not detect one Left and one Right hand (check handedness).")

    left_lms = hand_landmarks_list[left_idx]
    right_lms = hand_landmarks_list[right_idx]

    # Convert to (21, 3) arrays in image-normalized coordinates
    left_pts = np.array([[lm.x, lm.y, lm.z] for lm in left_lms], dtype=np.float32)
    right_pts = np.array([[lm.x, lm.y, lm.z] for lm in right_lms], dtype=np.float32)

    # ---- 3) Translation: use LEFT wrist as global origin ----
    WRIST = 0
    MIDDLE_MCP = 9
    INDEX_MCP = 5
    PINKY_MCP = 17

    left_wrist = left_pts[WRIST]
    left_pts_centered = left_pts - left_wrist
    right_pts_centered = right_pts - left_wrist  # right hand relative to left wrist

    # ---- 4) Scale: based on LEFT wrist -> LEFT middle MCP distance ----
    base_vec = left_pts_centered[MIDDLE_MCP]
    base_len = np.linalg.norm(base_vec)
    if base_len < 1e-6:
        base_len = 1e-6

    scale = scale_to / base_len
    left_pts_centered *= scale
    right_pts_centered *= scale

    # ---- 5) Orientation: build hand-centric frame from LEFT hand ----
    # e1: along left wrist -> left middle MCP
    e1 = left_pts_centered[MIDDLE_MCP]
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-6:
        e1_norm = 1e-6
    e1 /= e1_norm

    # across-palm direction: left index MCP -> left pinky MCP
    across = left_pts_centered[PINKY_MCP] - left_pts_centered[INDEX_MCP]
    across_norm = np.linalg.norm(across)
    if across_norm < 1e-6:
        across_norm = 1e-6
    across /= across_norm

    # palm normal
    normal = np.cross(e1, across)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-6:
        n_norm = 1e-6
    normal /= n_norm

    # e2: complete right-handed frame
    e2 = np.cross(normal, e1)
    e2_norm = np.linalg.norm(e2)
    if e2_norm < 1e-6:
        e2_norm = 1e-6
    e2 /= e2_norm

    # Rotation matrix (3x3) with columns [e1, e2, normal]
    R = np.stack([e1, e2, normal], axis=1)

    # ---- 6) Express BOTH hands in this LEFT-hand frame ----
    left_local = left_pts_centered @ R   # (21, 3)
    right_local = right_pts_centered @ R # (21, 3)

    # ---- 7) Stack into single (42, 3) array: Left first, Right second ----
    combined = np.vstack([left_local, right_local])  # (42, 3)

    return combined
