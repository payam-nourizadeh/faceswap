"""Evaluation metrics: identity preservation, FID, and blend seam quality"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from scipy import linalg


#####################################
# Identity verification
#####################################
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    #Row-wise cosine similarity between two (N, D) embedding arrays
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=1)


def interpolate_identity(emb_target: np.ndarray, emb_source: np.ndarray, alpha: float) -> np.ndarray:
    """Blend target and source embeddings and renormalise
    alpha=0 injects the target own identity (no swap); alpha=1 injects the
    full source identity. sweeping [0,1] to find the trade off.
    """
    blend = (1.0 - alpha) * emb_target + alpha * emb_source
    norm = np.linalg.norm(blend, axis=-1, keepdims=True) + 1e-8
    return blend / norm


def calibrate_threshold(impostor_scores: np.ndarray, far_target: float) -> float:
    #threshold at the (1 - far_target) quantile of the impostor distribution, calibrate

    if impostor_scores.size == 0:
        raise ValueError("need impostor scores to calibrate")
    if not 0.0 < far_target < 1.0:
        raise ValueError("far_target must be in (0, 1)")
    return float(np.quantile(impostor_scores, 1.0 - far_target))


@dataclass
class IdentityReport:
    threshold: float
    far_target: float
    preservation_rate: float     
    mean_similarity: float
    median_similarity: float


def identity_preservation(swap_vs_source: np.ndarray, impostor_scores: np.ndarray, far_target: float = 0.001,) -> IdentityReport:
    """Score identity preservation at a calibrated FAR 

    swap_vs_source: similarities between each swap and its source.
    impostor_scores: cross-identity similarities used only to set the threshold.
    """
    threshold = calibrate_threshold(impostor_scores, far_target)
    rate = float(np.mean(swap_vs_source >= threshold))
    return IdentityReport(
        threshold=threshold,
        far_target=far_target,
        preservation_rate=rate,
        mean_similarity=float(np.mean(swap_vs_source)),
        median_similarity=float(np.median(swap_vs_source)),
    )


#################################
# TO DOOOOOO //// Realism: Frechet Inception Distance
#################################
def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6,) -> float:
    #Frechet distance between two Gaussians (the FID formula). needs InceptionV3

    diff = mu1 - mu2
    sqrt_result = linalg.sqrtm(sigma1 @ sigma2)
    covmean = sqrt_result[0] if isinstance(sqrt_result, tuple) else sqrt_result
    if not np.isfinite(covmean).all():

        offset = np.eye(sigma1.shape[0]) * eps
        sqrt_result = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        covmean = sqrt_result[0] if isinstance(sqrt_result, tuple) else sqrt_result
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #Mean and covariance of an (N, D) feature matrix, for FID
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


#################################
# Blend quality
#################################
def seam_score(image_bgr: np.ndarray, mask: np.ndarray, band: int = 3) -> float:
    """Gradient ratio at the swap boundary; lower = cleaner blend.
    Computes mean gradient magnitude in a thin ring around the mask edge,
    divided by overall mean gradient. 1.0 ~ invisible seam, >1 ~ visible step.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    mask_bin = (mask > 0).astype(np.uint8)
    kernel = np.ones((band * 2 + 1, band * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask_bin, kernel)
    eroded = cv2.erode(mask_bin, kernel)
    boundary = (dilated - eroded).astype(bool)   # thin ring straddling the edge

    if boundary.sum() == 0:
        return 0.0
    boundary_grad = grad[boundary].mean()
    overall_grad = grad.mean() + 1e-6
    return float(boundary_grad / overall_grad)


