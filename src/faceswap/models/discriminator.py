"""Multi-scale PatchGAN discriminator (pix2pixHD-style)."""
from __future__ import annotations

import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """Single scale PatchGAN. Returns a list of feature maps. last entry is the patch-score map, earlier ones feed feature matching."""

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 4):
        super().__init__()
        blocks: list[list[nn.Module]] = [
            [nn.Conv2d(input_nc, ndf, 4, 2, 2), nn.LeakyReLU(0.2, inplace=True)]
        ]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            blocks.append([
                nn.Conv2d(nf_prev, nf, 4, 2, 2),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        nf_prev, nf = nf, min(nf * 2, 512)
        blocks.append([
            nn.Conv2d(nf_prev, nf, 4, 1, 2),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        blocks.append([nn.Conv2d(nf, 1, 4, 1, 2)])  # patch score map

        self.n_blocks = len(blocks)
        # Registered separately so forward() can read out intermediate features.
        for i, block in enumerate(blocks):
            setattr(self, f"block{i}", nn.Sequential(*block))

    def forward(self,x: torch.Tensor) -> list[torch.Tensor]:
        feats = [x]
        for i in range(self.n_blocks):
            feats.append(getattr(self, f"block{i}")(feats[-1]))
        return feats[1:]  # drop raw input. last entry is the score map


class MultiscaleDiscriminator(nn.Module):
    """Run num_D PatchGAN discriminators at cut resolutions."""

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 4,
        num_D: int = 3,
    ):
        super().__init__()
        self.num_D = num_D
        for i in range(num_D):
            setattr(self, f"disc{i}", NLayerDiscriminator(input_nc, ndf, n_layers))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        #Returns a list (per scale) of feature lists (per layer)
        results = []
        for i in range(self.num_D):
            results.append(getattr(self, f"disc{i}")(x))
            if i != self.num_D - 1:
                x = self.downsample(x)
        return results
    

