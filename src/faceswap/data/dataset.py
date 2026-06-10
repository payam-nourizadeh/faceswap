"""PyTorch dataset for face-swap training.

pairs each target with a source via PairSampler, and returns normalised tensors plus the same/cross-identity flag
the training loop needs.

Tensors is float32, (3, H, W), RGB, in [-1, 1] to match tanh generator output.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .pairing import PairSampler


########################
# Manifest indexing and splitting
########################

def load_manifest(path:str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_identity_index(entries: list[dict]) -> dict[str, list[int]]:
    """identity to list of positions in entries for what PairSampler needs"""
    index: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        index.setdefault(e["identity"], []).append(i)
    return index


def split_by_identity(manifest: dict, val_frac: float = 0.0, seed: int = 42) -> tuple[list[dict], list[dict]]:

    """Train/val split with disjoint identities, so val measures generalisation to new people rather than 
    memorisation. Defaults to 0.0 (use everything, identity preservation is measured separately"""
    entries = manifest["entries"]
    if val_frac <= 0.0:
        return entries, []
    
    ids = sorted({e["identity"] for e in entries})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    val_ids = set(ids[:n_val])
    train = [e for e in entries if e["identity"] not in val_ids]
    val = [e for e in entries if e["identity"] in val_ids]
    return train, val


########################
# Tensor helpers
########################
def image_to_tensor(image_rgb:np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()
    return t.div_(127.5).sub_(1.0)

def denormalize(t:torch.Tensor) -> torch.Tensor:
    return t.detach().clamp(-1, 1).add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)


##########################
# Dataset
##########################
class FaceSwapDataset(Dataset):
    """returns {target, source, same_identity, target_id, source_id}.
    args:
    target: as attribute; pose, expression, lighting);
    source: as identity; Embedding the source is left to the training loop (frozen ArcFace)
    """

    def __init__(self, entries: list[dict], image_size: int = 256, same_id_prob: float = 0.2, augment_flip: bool = True, sampler_rng: Optional[random.Random] = None,):
        if not entries:
            raise ValueError("FaceswapDataset received no entries")
        self.entries = entries
        self.image_size = image_size
        self.augment_flip = augment_flip
        index = build_identity_index(entries)
        self.sampler = PairSampler(index, same_id_prob=same_id_prob, rng=sampler_rng)

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, path: str, flip: bool) -> torch.Tensor:
        img= cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read aligned crop: {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_size and img.shape[0] != self.image_size:
            interp = cv2.INTER_AREA if img.shape[0] > self.image_size else cv2.INTER_LINEAR
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=interp)
        
        if flip:
            img = img[:, ::-1]
        return image_to_tensor(img)

    def __getitem__(self, i: int) -> dict:
        target_entry = self.entries[i]
        src_index, is_same = self.sampler.sample_source(target_entry["identity"], i)
        source_entry = self.entries[src_index]

        # One flip decision per sample, applied to both images so a same-image reconstruction stays consistent.
        flip = self.augment_flip and (random.random() < 0.5)

        return {
            "target": self._load(target_entry["aligned_path"], flip),
            "source": self._load(source_entry["aligned_path"], flip),
            "same_identity": is_same,
            "target_id": target_entry["identity"],
            "source_id": source_entry["identity"],
        }


def worker_init_fn(worker_id: int) -> None:
    # seed numpy per worker too (torch/random are already seeded per worker)
    import numpy as np
    import torch

    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
