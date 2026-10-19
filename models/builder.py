"""Model factory. To add a model: create models/<name>.py and add one entry below."""
import torch.nn as nn


def build_model(model_name, n_channels=3, n_classes=7, pretrained=True):
    name = model_name.lower()
    if name == "unet":
        from .unet import UNet
        return UNet(n_channels, n_classes)
    if name in ("unet_mobilenet", "mobilenet_unet"):
        from .unet_mobilenet import UNetMobileNetV3
        return UNetMobileNetV3(n_channels, n_classes, pretrained)
    if name in ("deeplabv3plus", "deeplabv3+"):
        from .deeplabv3plus import DeepLabV3Plus
        return DeepLabV3Plus(n_channels, n_classes, pretrained)
    # Planned baseline (add the file, then enable here):
    #   "segformer" -> models/segformer.py
    raise ValueError(f"Unknown model_name '{model_name}'. Available: {AVAILABLE_MODELS}")


AVAILABLE_MODELS = ["unet", "unet_mobilenet", "deeplabv3plus"]


def print_model_summary(model, model_name, batch_size, img_size, device):
    print("=" * 70)
    print(f" MODEL: {model_name.upper()}")
    print("=" * 70)
    try:
        from torchinfo import summary
        summary(model, input_size=(batch_size, 3, img_size, img_size), device=device.type)
    except ImportError:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters:     {total:,}\nTrainable parameters: {trainable:,}")
    print("=" * 70 + "\n")
