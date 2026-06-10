"""Head-pose estimation for attribute-preservation scoring"""
from __future__ import annotations

import cv2
import numpy as np


# Canonical 3D positions (mm, head-centric) for the 5 landmarks:
# left eye, right eye, nose tip, left mouth, right mouth.
# Approximate average-face geometry; exact values don't matter much for a
# relative target-vs-swap comparison since both use the same model.
_MODEL_3D = np.array(
    [
        [-30.0, 30.0, -30.0],   # left eye
        [30.0, 30.0, -30.0],    # right eye
        [0.0, 0.0, 0.0],        # nose tip (origin)
        [-25.0, -30.0, -30.0],  # left mouth corner
        [25.0, -30.0, -30.0],   # right mouth corner
    ],
    dtype=np.float64,
)


def _rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    from scipy.spatial.transform import Rotation
    return tuple(Rotation.from_matrix(R).as_euler("yxz", degrees=True))


def estimate_pose(landmarks_2d: np.ndarray, image_size: int) -> tuple[float, float, float]:
    #Estimate (yaw, pitch, roll) in degrees from 5 landmarks.
    
    if landmarks_2d.shape != (5, 2):
        raise ValueError(f"expected 5x2 landmarks, got {landmarks_2d.shape}")

    focal = float(image_size)
    center = (image_size / 2.0, image_size / 2.0)
    camera = np.array(
        [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64
    )
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_3D,
        landmarks_2d.astype(np.float64),
        camera,
        np.zeros((4, 1)),         # assume no lens distortion
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return (0.0, 0.0, 0.0)
    R, _ = cv2.Rodrigues(rvec)
    return _rotation_to_euler(R)


def pose_error(
    target_landmarks: np.ndarray, swap_landmarks: np.ndarray, image_size: int
) -> dict:
    #Per-axis and total absolute pose difference (degrees) between target and swap
    ty, tp, tr = estimate_pose(target_landmarks, image_size)
    sy, sp, sr = estimate_pose(swap_landmarks, image_size)
    dyaw, dpitch, droll = abs(sy - ty), abs(sp - tp), abs(sr - tr)
    return {
        "yaw_error": dyaw,
        "pitch_error": dpitch,
        "roll_error": droll,
        "total_error": dyaw + dpitch + droll,
    }

