"""
Device resolution. options: cuda, mps, cpu
"""

from __future__ import annotations
import os
import torch


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        # Allow unsupported MPS ops to degrade to CPU rather than raise.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")

    return torch.device("cpu")


def device_supports_amp(device: torch.device) -> bool:
    return device.type == "cuda"
