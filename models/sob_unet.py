"""SOB-UNet: the baseline U-Net MobileNetV3 plus two modules, each aimed at one
failure mode measured on this dataset (see error_budget.py).

BAG - Boundary Attention Gate   (this file, implemented)
    52% of the baseline's error is against the background: the predicted extent
    of a region does not match the annotated one. Efflorescence is the extreme
    case, with 100% of its error against background and none against any other
    class. BAG adds a small branch that predicts a damage-boundary map from the
    last decoder features, supervised by boundaries derived from the existing
    masks, and uses it as a residual attention gate so the decoder sharpens the
    region edge before the classification head.

OSH - Ordinal Stage Head        (next commit)
    The remaining 48% is confusion inside the severe-damage family, which is a
    physical sequence rather than a set of independent labels.

The gate is a residual multiplier, x * (1 + sigmoid(edge)), so at initialisation
the module is close to a no-op and cannot destroy the pretrained features.

During training the forward pass returns [seg_logits, edge_logit] concatenated
on the channel axis; in eval mode it returns the segmentation logits alone, so
evaluate.py, compare_runs.py and the metrics code work unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_mobilenet import UNetMobileNetV3


class BoundaryAttentionGate(nn.Module):
    """Predict a boundary map from decoder features and gate them with it."""

    def __init__(self, in_channels, mid_channels=16):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        edge_logits = self.edge(x)
        return x * (1.0 + torch.sigmoid(edge_logits)), edge_logits


class SOBUNet(UNetMobileNetV3):
    """Baseline U-Net MobileNetV3 with BAG (and later OSH) switched on per flag."""

    def __init__(self, n_channels=3, n_classes=7, pretrained=True, head="flat",
                 class_names=None, use_bag=True, use_osh=False):
        super().__init__(n_channels, n_classes, pretrained, head, class_names)
        if use_osh:
            raise NotImplementedError(
                "OSH (Ordinal Stage Head) is not implemented yet; run with --no-use_osh.")
        self.use_bag = use_bag
        self.use_osh = use_osh
        if use_bag:
            self.bag = BoundaryAttentionGate(16)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        edge_logits = None
        if self.use_bag:
            x, edge_logits = self.bag(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        seg = self.outc(x)

        if self.training and edge_logits is not None:
            edge_logits = F.interpolate(edge_logits, size=seg.shape[-2:],
                                        mode="bilinear", align_corners=True)
            return torch.cat([seg, edge_logits], dim=1)
        return seg
