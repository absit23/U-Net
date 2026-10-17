import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCombinedLoss(nn.Module):
    """Weighted cross-entropy + soft Dice. logits [B,C,H,W], targets [B,H,W] (IDs 0..C-1)."""

    def __init__(self, num_classes, weight_ce=1.0, weight_dice=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        if logits.ndim != 4 or targets.ndim != 3:
            raise ValueError(f"Expected logits [B,C,H,W] and masks [B,H,W], got {tuple(logits.shape)} / {tuple(targets.shape)}")
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"Model outputs {logits.shape[1]} classes, loss expects {self.num_classes}.")
        if logits.shape[2:] != targets.shape[1:]:
            raise ValueError("Logit/mask spatial size mismatch. Check resizing.")

        targets = targets.long()
        if targets.numel() and (targets.min() < 0 or targets.max() >= self.num_classes):
            raise ValueError(
                f"Mask IDs must be in [0, {self.num_classes - 1}], "
                f"got min={targets.min().item()}, max={targets.max().item()}. Check --variant."
            )

        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).to(probs.dtype)
        dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dim=dims)
        card = torch.sum(probs + one_hot, dim=dims)
        dice_loss = 1.0 - ((2.0 * inter + 1e-7) / (card + 1e-7)).mean()

        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
