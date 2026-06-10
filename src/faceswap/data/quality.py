"""Quality control for aligned faces.
checking blurry, low-res, or steeply off-angle.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .alignment import AlignedFace


def blur_score(image_bgr: np.ndarray) -> float:
    # var of the Laplacian
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def interocular_distance(landmarks: np.ndarray) -> float:
    # Eye-to-eye pixel distance, small values= mostly interpolated.
    return float(np.linalg.norm(landmarks[0] - landmarks[1]))

def frontality(landmarks: np.ndarray) -> float:
    # rough symmetry score in [0, 1], ~1 frontal, lower for profiles.

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_left_d = abs(nose[0] - left_eye[0])
    eye_right_d = abs(right_eye[0] - nose[0])
    mouth_left_d = abs(nose[0] - left_mouth[0])
    mouth_right_d = abs(right_mouth[0] - nose[0])

    def _symmetry(a: float, b: float) -> float:
        denom = max(a, b)
        return float(min(a, b) / denom) if denom > 1e-6 else 0.0

    return 0.5 * (_symmetry(eye_left_d, eye_right_d) + _symmetry(mouth_left_d, mouth_right_d))


@dataclass
class QualityThresholds:
    # LFW isnt big, so we keep as much as we can. only cut the clearly unusable tail.
    min_det_score: float = 0.20
    min_blur: float = 8.0
    min_interocular: float = 15.0
    min_frontality: float = 0.35


@dataclass
class QualityReport:
    det_score: float
    blur: float
    interocular: float
    frontality: float
    accepted: bool
    reasons: list[str]


def assess(face: AlignedFace, thr: QualityThresholds) -> QualityReport:
    #Score one aligned face and decide whether to keep it

    blur = blur_score(face.image)
    iod = interocular_distance(face.landmarks)
    front = frontality(face.landmarks)

    reasons: list[str] = []
    if face.det_score < thr.min_det_score:
        reasons.append("low_det_score")
    if blur < thr.min_blur:
        reasons.append("too_blurry")
    if iod < thr.min_interocular:
        reasons.append("too_small")
    if front < thr.min_frontality:
        reasons.append("too_off_angle")

    return QualityReport(
        det_score=face.det_score,
        blur=blur,
        interocular=iod,
        frontality=front,
        accepted=len(reasons) == 0,
        reasons=reasons,
    )



