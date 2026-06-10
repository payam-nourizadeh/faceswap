"""
Face detection and alignment

Everything downstream (identity encoder, generator) assumes a face crop where the eyes/nose/mouth sit in the same place every time. that's what
lets one model work across different faces. So this runs first.

For each image image: detect 5 landmarks, estimate a similarity transform onto a fixed template, warp into a square crop. We keep both the forward transform and its
inverse; the inverse is what pastes a swapped face back into the original photo later.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import cv2
import numpy as np


# ArcFace's canonical 5 points for a 112x112 crop: left eye, right eye, nose,
# left mouth, right mouth. Aligning to these makes our crops match what the
# pretrained encoder expects. Scaled linearly for larger crops.
_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class AlignedFace:
    """output of aligning one face.

    image: aligned BGR crop (size, size, 3) uint8
    landmarks: the 5 detected points in original-image coords
    transform, inverse_transform: 2x3 affine, original and aligned vice versa. The inverse is used at inference to paste the generated face back
    bbox: detected box [x1,y1,x2,y2] in original coords (can be None).
    det_score: detector confidence, carried through for the quality gate.
    """

    image: np.ndarray
    landmarks: np.ndarray
    transform: np.ndarray
    inverse_transform: np.ndarray
    bbox: Optional[np.ndarray]
    det_score: float


# placeholder for using different detecors; important to test the model on another face detector, i.e, r50 for training and r100 for evaluate
class FaceDetector(Protocol):
    # Anything with detect(image_bgr) -> list of dicts with 'kps' (5x2) and
    # optionally 'bbox'/'det_score' works. InsightFaceDetector below adapts
    # InsightFace to this.
    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        ...


def get_template(output_size: int) -> np.ndarray:
    # Scale the 112px template to the crop size. we use 256px, the encoder resizes internally so the larger crop just gives the generator
    return _ARCFACE_TEMPLATE_112 * (output_size / 112.0)


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Least-squares similarity transform (Umeyama 1991), src to dst.

    Returns a 2x3 [sR | t]: rotation, single uniform scale, translation (4 DOF)
    can also use skimage.SimilarityTransform / InsightFace compute internally.
    done manually here to avoid a deprecated API and keep it explicit. something I had issues with in the past
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    n = src.shape[0]

    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_demean = src - mean_src
    dst_demean = dst - mean_dst

    cov = dst_demean.T @ src_demean / n
    U, S, Vt = np.linalg.svd(cov)

    # Guard against the rotation flipping into a reflection on near-degenerate
    # point sets.
    d = np.ones(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1.0

    R = U @ np.diag(d) @ Vt
    var_src = src_demean.var(axis=0).sum()
    scale = (S * d).sum() / var_src
    t = mean_dst - scale * (R @ mean_src)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = (scale * R).astype(np.float32)
    M[:, 2] = t.astype(np.float32)
    return M


def estimate_norm(landmarks: np.ndarray, output_size: int) -> np.ndarray:
    """2x3 affine mapping detected landmarks to canonical template """
    if landmarks.shape !=(5, 2):
        raise ValueError(f"Expected 5x2 landmarks, got {landmarks.shape}")

    template = get_template(output_size)
    return _umeyama_similarity(landmarks, template)


class FaceAligner:
    # Detect and align faces to the canonical template

    def __init__(self, detector: FaceDetector, output_size: int = 256):
        self.detector = detector
        self.output_size = output_size

    def align_from_landmarks(self, image_bgr: np.ndarray, landmarks: np.ndarray, bbox: Optional[np.ndarray] = None,
                                det_score: float = 1.0,) -> AlignedFace:
        
        #Align given precomputed landmarks (no detection)
        M = estimate_norm(landmarks, self.output_size)
        aligned = cv2.warpAffine(image_bgr, M, (self.output_size, self.output_size), borderValue=0.0,)

        inverse_M = cv2.invertAffineTransform(M)
        
        return AlignedFace(
            image=aligned,
            landmarks=landmarks,
            transform=M,
            inverse_transform=inverse_M,
            bbox=bbox,
            det_score=det_score,
        )

    def align(self, image_bgr: np.ndarray) -> Optional[AlignedFace]:
        #Detect the largest face and align it. None if nothing is detected        
        
        faces = self.detector.detect(image_bgr)
        if not faces:
            return None

        # largest face by box area when we have boxes, else the first one
        def _area(face: dict) -> float:
            box=face.get("bbox")
            if box is None:
                return 0.0
            return float((box[2] - box[0]) * (box[3] - box[1]))

        face = max(faces, key=_area) if faces[0].get("bbox") is not None else faces[0]

        return self.align_from_landmarks(
            image_bgr,
            np.asarray(face["kps"], dtype=np.float32),
            bbox=face.get("bbox"),
            det_score=float(face.get("det_score", 1.0)),
        )

    def paste_back(self, original_bgr: np.ndarray, swapped_face:np.ndarray, aligned: AlignedFace,) -> np.ndarray:
        # Hard-edged paste using the inverse transform. The soft blending (feather/Poisson) live in the inference module. this just keeps the geometric inverse next to the forward transform
        h, w = original_bgr.shape[:2]
        warped_back = cv2.warpAffine(swapped_face, aligned.inverse_transform, (w, h))
        mask = np.full(swapped_face.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpAffine(mask, aligned.inverse_transform, (w, h))
        out = original_bgr.copy()
        out[warped_mask > 0] = warped_back[warped_mask > 0]
        return out


class InsightFaceDetector:
    """Adapts InsightFace's FaceAnalysis to the FaceDetector protocol.
    Gives a RetinaFace detector + 5-point landmarks. imported lazily. the model packs are non-commercial research licensed, fine for this eval.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640, providers=None):
        from insightface.app import FaceAnalysis  # lazy import

        if providers is None:
            # providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"] # conflict when training on nvidia and running on mps
            providers = ["CPUExecutionProvider"]  # remove CoreML entirely, switch to above if running on nvidia 
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        faces = self.app.get(image_bgr)
        
        return [
            {
                "kps": f.kps,
                "bbox": f.bbox,
                "det_score": getattr(f, "det_score", 1.0),
            }
            for f in faces
        ]
    
    