"""DeepLabv3+ baseline (via segmentation_models_pytorch), ImageNet-pretrained encoder."""
import segmentation_models_pytorch as smp
import torch.nn as nn


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_channels=3, n_classes=8, pretrained=True, encoder="resnet34"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.net = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=n_channels,
            classes=n_classes,
        )

    def forward(self, x):
        return self.net(x)
