"""U-Net with an ImageNet-pretrained MobileNetV3-Large encoder (~3.4M params)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .unet_parts import DoubleConv, OutConv


class UpBlock(nn.Module):
    """Bilinear upsample, concatenate skip, DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y, diff_x = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNetMobileNetV3(nn.Module):
    def __init__(self, n_channels=3, n_classes=7, pretrained=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        features = tvm.mobilenet_v3_large(weights=weights).features
        if n_channels != 3:
            features[0][0] = nn.Conv2d(n_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)

        # Encoder stages (channels, resolution)
        self.inc = features[0:2]      # 16,  H/2
        self.down1 = features[2:4]    # 24,  H/4
        self.down2 = features[4:7]    # 40,  H/8
        self.down3 = features[7:13]   # 112, H/16
        self.down4 = features[13:16]  # 160, H/32 (bottleneck)

        # Decoder (in = upsampled channels + skip channels)
        self.up1 = UpBlock(160 + 112, 128)
        self.up2 = UpBlock(128 + 40, 64)
        self.up3 = UpBlock(64 + 24, 32)
        self.up4 = UpBlock(32 + 16, 16)
        self.outc = OutConv(16, n_classes)

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
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.outc(x)
