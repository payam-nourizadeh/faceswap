"""Blend a swapped aligned face back into the original image.

Two methods:
  alpha: feathered ellipse mask, fast and robust (defualt).
  poisson: gradient-domain (seamlessClone), corrects lighting/skin-tone mismatch rather than just hiding the edge. slower
"""
from __future__ import annotations

import cv2
import numpy as np

from .data.alignment import AlignedFace


def feathered_ellipse_mask(size: int, feather_frac: float = 0.12) -> np.ndarray:
    #Soft elliptical mask (float32 [0,1]) covering the central face region
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    axes = (int(size * 0.42), int(size * 0.52))   # slightly taller than wide
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    k = max(3, int(size * feather_frac) | 1)       # odd kernel
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)

def _warp_to_original(
    swapped_aligned: np.ndarray, inverse_M: np.ndarray, out_hw: tuple[int, int]
) -> np.ndarray:
    h, w = out_hw
    return cv2.warpAffine(swapped_aligned, inverse_M, (w, h))


def alpha_paste(original_bgr: np.ndarray, swapped_aligned: np.ndarray, inverse_M: np.ndarray, mask: np.ndarray,) -> np.ndarray:
    #Feather alpha: swapped face into the original frame
    h, w = original_bgr.shape[:2]
    warped = _warp_to_original(swapped_aligned, inverse_M, (h, w)).astype(np.float32)
    warped_mask = _warp_to_original(mask, inverse_M, (h, w))
    warped_mask = np.clip(warped_mask, 0.0, 1.0)[..., None]   # HxWx1 for broadcast
    out = original_bgr.astype(np.float32) * (1.0 - warped_mask) + warped * warped_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def poisson_paste(original_bgr: np.ndarray,swapped_aligned: np.ndarray, inverse_M: np.ndarray, mask: np.ndarray,) -> np.ndarray:
    #Poisson blend; falls back to alpha if the warped mask is empty; face outside frame
    h, w = original_bgr.shape[:2]
    warped = _warp_to_original(swapped_aligned, inverse_M, (h, w))
    warped_mask = _warp_to_original(mask, inverse_M, (h, w))
    binary = (warped_mask > 0.5).astype(np.uint8) * 255

    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        return alpha_paste(original_bgr, swapped_aligned, inverse_M, mask)
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    return cv2.seamlessClone(warped, original_bgr, binary, center, cv2.NORMAL_CLONE)


def blend_back(original_bgr: np.ndarray, swapped_aligned: np.ndarray, aligned: AlignedFace,method: str = "alpha", mask: np.ndarray | None = None) -> np.ndarray:
    #Place a swapped aligned face into the original frame

    if mask is None:
        mask = feathered_ellipse_mask(swapped_aligned.shape[0])
    if method == "poisson":
        return poisson_paste(original_bgr, swapped_aligned, aligned.inverse_transform, mask)
    if method == "alpha":
        return alpha_paste(original_bgr, swapped_aligned, aligned.inverse_transform, mask)
    raise ValueError(f"unknown blend method {method!r}; use 'alpha' or 'poisson'")





