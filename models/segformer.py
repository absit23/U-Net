"""SegFormer (MiT encoder) baseline via segmentation_models_pytorch.

segformer     -> MiT-B2 (about 25M params)
segformer_b0  -> MiT-B0 (about 3.7M params, similar size to the MobileNetV3 U-Net)
"""
import segmentation_models_pytorch as smp
import torch.nn as nn


class SegFormer(nn.Module):
    def __init__(self, n_channels=3, n_classes=8, pretrained=True, encoder="mit_b2"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.net = smp.Segformer(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=n_channels,
            classes=n_classes,
        )

    def forward(self, x):
        return self.net(x)
