"""PyTorch architectures for every model we train ourselves.

Torch is only needed for training/export. The runtime app loads the exported ONNX files
through ONNX Runtime (see saree_ai/inference).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import SEG_CLASSES, PLEAT_QUALITY, DRAPE_STAGES, KEYPOINTS
from .features import FEATURE_DIM


# ------------------------------------------------------------------ Model B: segmentation
class _ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    """~0.5M-param U-Net, trained entirely from scratch. 'lite' profile (CPU friendly)."""

    def __init__(self, num_classes=len(SEG_CLASSES), base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1 = nn.Sequential(_ConvBNReLU(3, c[0]), _ConvBNReLU(c[0], c[0]))
        self.e2 = nn.Sequential(_ConvBNReLU(c[0], c[1], 2), _ConvBNReLU(c[1], c[1]))
        self.e3 = nn.Sequential(_ConvBNReLU(c[1], c[2], 2), _ConvBNReLU(c[2], c[2]))
        self.e4 = nn.Sequential(_ConvBNReLU(c[2], c[3], 2), _ConvBNReLU(c[3], c[3]))
        self.d3 = nn.Sequential(_ConvBNReLU(c[3] + c[2], c[2]), _ConvBNReLU(c[2], c[2]))
        self.d2 = nn.Sequential(_ConvBNReLU(c[2] + c[1], c[1]), _ConvBNReLU(c[1], c[1]))
        self.d1 = nn.Sequential(_ConvBNReLU(c[1] + c[0], c[0]), _ConvBNReLU(c[0], c[0]))
        self.head = nn.Conv2d(c[0], num_classes, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d3 = self.d3(torch.cat([self._up(e4, e3), e3], 1))
        d2 = self.d2(torch.cat([self._up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([self._up(d2, e1), e1], 1))
        return self.head(d1)


class LRASPPSeg(nn.Module):
    """LR-ASPP + MobileNetV3-Large (torchvision). 'full' profile.

    Backbone optionally initialised from torchvision's ImageNet-1k weights (BSD-3 code;
    see docs/DEPENDENCIES.md). The segmentation head is always trained by us from scratch.
    """

    def __init__(self, num_classes=len(SEG_CLASSES), imagenet_backbone=True):
        super().__init__()
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        from torchvision.models import MobileNet_V3_Large_Weights
        wb = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if imagenet_backbone else None
        self.net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=wb, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)["out"]


def build_segmentation(arch: str, /, **kw) -> nn.Module:
    kw.pop("arch", None)
    if arch == "tiny_unet":
        return TinyUNet(**{k: v for k, v in kw.items() if k in ("base",)})
    if arch == "lraspp_mobilenet_v3":
        return LRASPPSeg(imagenet_backbone=kw.get("imagenet_backbone", True))
    raise ValueError(f"unknown segmentation arch {arch}")


# ------------------------------------------------------------------ Model C: pleats
class PleatNet(nn.Module):
    """Multi-head CNN on a pleat-region crop, trained from scratch.

    Outputs (all logits / raw):
      count      (B,1)  expected number of visible pleats (trained with smooth-L1)
      quality    (B,4)  PLEAT_QUALITY logits
      secured    (B,1)  logit: pleat stack tucked at waist
      visible    (B,1)  logit: a pleat stack is actually present in the crop
    """

    def __init__(self, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _ConvBNReLU(3, w, 2), _ConvBNReLU(w, w),
            _ConvBNReLU(w, 2 * w, 2), _ConvBNReLU(2 * w, 2 * w),
            _ConvBNReLU(2 * w, 4 * w, 2), _ConvBNReLU(4 * w, 4 * w),
            _ConvBNReLU(4 * w, 8 * w, 2),
        )
        # Pleats are mostly vertical periodic structure: pool rows away but keep a column profile.
        self.col_pool = nn.AdaptiveAvgPool2d((1, 8))
        d = 8 * w * 8
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.count = nn.Linear(256, 1)
        self.quality = nn.Linear(256, len(PLEAT_QUALITY))
        self.secured = nn.Linear(256, 1)
        self.visible = nn.Linear(256, 1)

    def forward(self, x):
        h = self.trunk(self.col_pool(self.features(x)))
        return self.count(h), self.quality(h), self.secured(h), self.visible(h)


# ------------------------------------------------------------------ Model D: draping stage
class StageMLP(nn.Module):
    """Per-frame stage classifier on the interpretable feature vector (saree_ai.features).
    Temporal consistency is added at inference by a tutorial-order HMM filter."""

    def __init__(self, in_dim=FEATURE_DIM, hidden=64, num_classes=len(DRAPE_STAGES), dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ Model A': our own pose net
class PoseHeatmapNet(nn.Module):
    """SimpleBaseline-style top-down heatmap pose model (Xiao et al. 2018 design),
    MobileNetV3-Large backbone. Input: person crop 256x192. Output: 17 heatmaps at 1/4 res.

    This is the independently maintained replacement for the pretrained baseline. It can be
    trained on our own landmark annotations, on COCO-Keypoints (annotations CC BY 4.0), or by
    distillation from the baseline on our own consented videos.
    """

    def __init__(self, num_kp=len(KEYPOINTS), imagenet_backbone=True):
        super().__init__()
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        wb = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if imagenet_backbone else None
        self.backbone = mobilenet_v3_large(weights=wb).features  # stride 32, 960 ch
        layers, cin = [], 960
        for _ in range(3):
            layers += [nn.ConvTranspose2d(cin, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
            cin = 128
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv2d(128, num_kp, 1)

    def forward(self, x):
        return self.head(self.deconv(self.backbone(x)))
