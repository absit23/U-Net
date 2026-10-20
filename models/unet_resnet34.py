"""U-Net with an ImageNet-pretrained ResNet34 encoder (control for decoder-vs-encoder size)."""
import segmentation_models_pytorch as smp
import torch.nn as nn


class UNetResNet34(nn.Module):
    def __init__(self, n_channels=3, n_classes=8, pretrained=True, encoder="resnet34"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=n_channels,
            classes=n_classes,
        )

    def forward(self, x):
        return self.net(x)
