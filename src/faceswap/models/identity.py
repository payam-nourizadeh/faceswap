"""Identity encoder: frozen ArcFace network -> L2 normalised identity embeddings.

module is pinned to eval(), but the forward pass is not wrapped in no_grad
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(inp: int, outp: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inp, outp, 3, stride, 1, bias=False)


def _conv1x1(inp: int, outp: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inp, outp, 1, stride, bias=False)


class IBasicBlock(nn.Module):
    """BN-first residual block by Arcface Iresnet """

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """Arcface Iresnet. Input: 112x112 rgb. Output: num_features embedding."""

    fc_scale = 7 * 7  # spatial size after 4 stride-2 stages from 112px

    def __init__(self, layers: list[int], dropout:float = 0.0, num_features: int = 512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes, eps=1e-5),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.features(x)


def iresnet50(**kw) -> IResNet:
    return IResNet([3, 4, 14, 3], **kw)


def iresnet100(**kw) -> IResNet:
    return IResNet([3, 13, 30, 3], **kw)


_BACKBONES = {"r50": iresnet50, "r100": iresnet100}


class IdentityEncoder(nn.Module):
    #Frozen Arcfac wrapper: face crop to unit-norm embedding

    def __init__(
        self,
        backbone: str = "r50",
        weights_path: Optional[str | Path] = None,
        input_size: int = 112,
    ):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; choose from {list(_BACKBONES)}")
        self.net = _BACKBONES[backbone]()
        self.input_size = input_size

        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            # accept either a raw state_dict or a checkpoint dict
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            if missing or unexpected:
                # BAM BAM!
                print(f"[IdentityEncoder] loaded with missing={len(missing)} "
                      f"unexpected={len(unexpected)} keys")

        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def train(self, mode: bool = True):
        # Override so model.train() on the parent doesnt unfreeze BN
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (B,3,H,W) RGB in [-1,1]. Returns (B, num_features), L2-normalised.

        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return F.normalize(self.net(x), dim=1)
    






