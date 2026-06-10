"""SimSwap-style face-swap generator.

target image -> encoder -> AdaIN residual blocks, identity injected here -> decoder -> swapped image

Upsample uses bilinear-resize-then-conv to avoid artefacts from transposed convs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ApplyStyle(nn.Module):
    #AdaIN modulation predict per-channel scale & bias from the identity vector

    def __init__(self, latent_dim: int,channels: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, channels * 2)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        style = self.linear(latent).unsqueeze(2).unsqueeze(3)  # (B, 2C, 1, 1)
        gamma, beta = style.chunk(2, dim=1)
        # (gamma + 1) initialise near identity so early training is stable.
        return x * (gamma + 1.0) + beta


class AdaINResBlock(nn.Module):
    #Residual block with identity normalisation

    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.style1 = ApplyStyle(latent_dim, channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.style2 = ApplyStyle(latent_dim, channels)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        y = self.style1(self.norm1(self.conv1(x)), latent)
        y = self.act(y)
        y = self.style2(self.norm2(self.conv2(y)), latent)
        return x + y


class Generator(nn.Module):
    """Identity-agnostic faceswap generator."""

    def __init__(self, img_channels: int = 3, base_channels: int = 64,id_embed_dim: int = 512, 
                 num_id_blocks: int = 9, num_scales: int = 3,):
        super().__init__()

        # Stem at full resolution.
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channels, base_channels, 7),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder: stride-2 downsamples, doubling channels each time.
        down = []
        c = base_channels
        for _ in range(num_scales):
            down += [
                nn.Conv2d(c, c * 2, 3, 2, 1),
                nn.BatchNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2
        self.encoder = nn.Sequential(*down)
        self.bottleneck_channels = c  # base * 2**num_scales (e.g. 512)

        self.id_blocks = nn.ModuleList(
            [AdaINResBlock(c, id_embed_dim) for _ in range(num_id_blocks)]
        )

        # Decoder: bilinear upsample + conv, halving channels each time.
        up = []
        for _ in range(num_scales):
            up += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c, c // 2, 3, 1, 1),
                nn.BatchNorm2d(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2
        self.decoder = nn.Sequential(*up)

        # tanh -> [-1, 1] to match dataset tensors.
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, img_channels, 7),
            nn.Tanh(),
        )

    def forward(self, target_img: torch.Tensor, id_embed: torch.Tensor) -> torch.Tensor:
        x = self.stem(target_img)
        x = self.encoder(x)
        for block in self.id_blocks:
            x = block(x, id_embed)
        x = self.decoder(x)
        return self.head(x)
    


