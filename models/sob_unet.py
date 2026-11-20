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


class OrdinalStageHead(nn.Module):
    """Predict how far along the damage sequence a pixel is.

    For a chain of K ordered stages the head emits K-1 logits per pixel, the
    k-th standing for P(stage >= k) (a CORAL-style cumulative encoding). It is
    an auxiliary head: it shapes the shared decoder features during training and
    is dropped at inference, so the segmentation output is unchanged.
    """

    def __init__(self, in_channels, n_stages, mid_channels=16):
        super().__init__()
        self.n_stages = n_stages
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, max(n_stages - 1, 1), kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class SOBUNet(UNetMobileNetV3):
    """Baseline U-Net MobileNetV3 with BAG and/or OSH switched on per flag."""

    def __init__(self, n_channels=3, n_classes=7, pretrained=True, head="flat",
                 class_names=None, use_bag=True, use_osh=False, n_stages=4):
        super().__init__(n_channels, n_classes, pretrained, head, class_names)
        self.use_bag = use_bag
        self.use_osh = use_osh
        self.n_stages = n_stages
        if use_bag:
            self.bag = BoundaryAttentionGate(16)
        if use_osh:
            self.osh = OrdinalStageHead(16, n_stages)

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

        ord_logits = self.osh(x) if self.use_osh else None

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        seg = self.outc(x)

        if not self.training or (edge_logits is None and ord_logits is None):
            return seg

        # training: append the auxiliary logits, in a fixed order the loss knows
        parts = [seg]
        size = seg.shape[-2:]
        if edge_logits is not None:
            parts.append(F.interpolate(edge_logits, size=size,
                                       mode="bilinear", align_corners=True))
        if ord_logits is not None:
            parts.append(F.interpolate(ord_logits, size=size,
                                       mode="bilinear", align_corners=True))
        return torch.cat(parts, dim=1)
