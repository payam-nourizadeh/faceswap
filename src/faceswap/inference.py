"""Inference pipeline: detect -> align -> embed source -> generate -> blend back."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
import torch

# from functions
from .blend import blend_back
from .data.alignment import FaceAligner
from .data.dataset import denormalize, image_to_tensor
from .models import Generator, IdentityEncoder
from .utils.device import get_device

log = logging.getLogger(__name__)


class FaceSwapper:
    def __init__(self, generator: torch.nn.Module, identity_encoder: torch.nn.Module, aligner: FaceAligner,
                    device: torch.device, image_size: int = 256, blend_method: str = "alpha",):
        self.generator = generator.eval().to(device)
        self.identity_encoder = identity_encoder.eval().to(device)
        self.aligner = aligner
        self.device = device
        self.image_size = image_size
        self.blend_method = blend_method

    # core steps
    def _to_tensor(self, aligned_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        return image_to_tensor(rgb).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_identity(self, source_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """Detect + align the source face and return its identity embedding."""
        aligned = self.aligner.align(source_bgr)
        if aligned is None:
            return None
        return self.identity_encoder(self._to_tensor(aligned.image))

    @torch.no_grad()
    def _swap_one(self, aligned_bgr: np.ndarray, identity: torch.Tensor) -> np.ndarray:
        """Run the generator on one aligned target face -> aligned swapped BGR."""
        swapped = self.generator(self._to_tensor(aligned_bgr), identity)[0]
        rgb = denormalize(swapped).permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # TO DO:  public API
    @torch.no_grad()
    def swap_image(
        self, source_bgr: np.ndarray, target_bgr: np.ndarray, all_faces: bool = False, strength: float = 1.0) -> np.ndarray:
        #Swap the source identity into the target image
        
        identity = self.embed_identity(source_bgr)
        if identity is None:
            log.warning("no face detected in the source image; returning target unchanged")
            return target_bgr

        faces = self.aligner.detector.detect(target_bgr)
        if not faces:
            log.warning("no face detected in the target image; returning it unchanged")
            return target_bgr
        if not all_faces:
            faces = [max(faces, key=lambda f: _bbox_area(f.get("bbox")))]

        result = target_bgr.copy()
        for face in faces:
            aligned = self.aligner.align_from_landmarks(
                target_bgr,
                np.asarray(face["kps"], dtype=np.float32),
                bbox=face.get("bbox"),
                det_score=float(face.get("det_score", 1.0)),
            )

            # blend toward target identity if strength < 1.0
            if strength < 1.0:
                import torch.nn.functional as F
                tgt_id = self.identity_encoder(self._to_tensor(aligned.image))
                inj = F.normalize(strength * identity + (1 - strength) * tgt_id, dim=1)
            else:
                inj = identity

            swapped_aligned = self._swap_one(aligned.image, inj)
            result = blend_back(result, swapped_aligned, aligned, method=self.blend_method)

        return result

    def warmup(self) -> None:
        # One dummy pass so the first real request isn't slowed by lazy allocation.
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        with torch.no_grad():
            ident = self.identity_encoder(self._to_tensor(dummy))
            self.generator(self._to_tensor(dummy), ident)
        log.info("warmup complete on %s", self.device)


def _bbox_area(box) -> float:
    if box is None:
        return 0.0
    return float((box[2] - box[0]) * (box[3] - box[1]))


#####################################
# Construction from config + checkpoint
#####################################
def load_swapper(cfg: dict, checkpoint: str | Path, detector: Optional[Callable] = None, blend_method: str = "alpha") -> FaceSwapper:
    #Build a FaceSwapper from config + checkpoint.

    m = cfg["model"]
    device = get_device(cfg["run"]["device"])

    generator = Generator(
        base_channels=m["generator_channels"],
        id_embed_dim=m["id_embed_dim"],
        num_id_blocks=m["num_id_blocks"],
        num_scales=m["num_scales"],
    )
    ckpt = torch.load(checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])

    encoder = IdentityEncoder(
        backbone=m["id_encoder"]["backbone"], weights_path=m["id_encoder"].get("weights")
    )

    if detector is None:
        from .data.alignment import InsightFaceDetector

        detector = InsightFaceDetector()
    aligner = FaceAligner(detector=detector, output_size=cfg["data"]["image_size"])

    swapper = FaceSwapper(
        generator, encoder, aligner, device,
        image_size=cfg["data"]["image_size"], blend_method=blend_method,
    )
    swapper.warmup()
    return swapper

