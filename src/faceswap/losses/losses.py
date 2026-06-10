"""Loss functions for the face-swap training loop."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def identity_loss(emb_fake: torch.Tensor, emb_source: torch.Tensor) -> torch.Tensor:
    # Both embeddings are already L2 normalised, so cosine is just a dot product.
    cosine = (emb_fake * emb_source).sum(dim=1)
    return (1.0 - cosine).mean()


def reconstruction_loss(fake: torch.Tensor, target: torch.Tensor, same_mask:torch.Tensor) -> torch.Tensor:
    # Only same-identity samples have pixel ground truth. cross-identity steps must not contribute. Returns 0 if the batch has no same-id samples.
    per_sample = (fake - target).abs().flatten(1).mean(dim=1)  # (B,)
    mask = same_mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (per_sample * mask).sum() / denom


class HingeGANLoss:
    #Hinge loss over a multiscale discriminator.

    @staticmethod
    def _scores(outputs:list[list[torch.Tensor]]) -> list[torch.Tensor]:
        return [scale[-1] for scale in outputs]

    def d_loss(self,real_outputs: list[list[torch.Tensor]], fake_outputs: list[list[torch.Tensor]],) -> torch.Tensor:
        
        loss = 0.0
        scores_real = self._scores(real_outputs)
        scores_fake = self._scores(fake_outputs)
        for r, f in zip(scores_real, scores_fake):
            loss = loss + torch.relu(1.0 - r).mean() + torch.relu(1.0 + f).mean()
        return loss / len(scores_real)

    def g_loss(self, fake_outputs: list[list[torch.Tensor]]) -> torch.Tensor:

        scores_fake = self._scores(fake_outputs)
        loss = 0.0
        for f in scores_fake:
            loss = loss - f.mean()
        return loss / len(scores_fake)


class FeatureMatchingLoss(nn.Module):
    """L1 between discrim intermediate features of fake vs real, target.

    Wcoming from SimSwap, only match the deepest weak_layers intermediate layers forceing low-level texture similarity. fight the identity change. generator only.
    """

    def __init__(self, weak: bool = True, weak_layers: int = 2):
        super().__init__()
        self.weak = weak
        self.weak_layers = weak_layers

    def forward(self,fake_outputs: list[list[torch.Tensor]], real_outputs: list[list[torch.Tensor]],) -> torch.Tensor:
        loss = 0.0
        count = 0
        for fake_scale, real_scale in zip(fake_outputs, real_outputs):
            inter_fake = fake_scale[:-1]   # exclude the score map
            inter_real = real_scale[:-1]
            if self.weak:
                inter_fake = inter_fake[-self.weak_layers:]
                inter_real = inter_real[-self.weak_layers:]
            for ff, rf in zip(inter_fake, inter_real):
                loss = loss + (ff - rf.detach()).abs().mean()
                count += 1
        if count == 0:
            return torch.zeros((), device=fake_outputs[0][-1].device)
        return loss / count
    
