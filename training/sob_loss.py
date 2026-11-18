"""Loss for SOB-UNet: the baseline segmentation loss plus a boundary term.

The boundary target is derived from the mask itself, so no new annotation is
needed: a pixel is a boundary pixel when the label is not constant in its
neighbourhood. That covers both the damage/background edge (the dominant error
in the budget) and the edge between two damage classes.

Boundary pixels are a small fraction of the image, so the BCE is weighted by
the negative/positive ratio of the batch, capped so one almost-empty mask
cannot blow the term up.

The same object is used for validation, where the model returns segmentation
logits only; the boundary term is then simply absent, and the printed
validation loss stays comparable with the baseline's.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import WeightedCombinedLoss


class SOBLoss(nn.Module):
    def __init__(self, num_classes, weight_ce=1.0, weight_dice=1.0, class_weights=None,
                 weight_edge=0.3, edge_width=1, pos_weight_cap=20.0):
        super().__init__()
        self.num_classes = num_classes
        self.weight_edge = weight_edge
        self.edge_width = int(edge_width)
        self.pos_weight_cap = pos_weight_cap
        self.base = WeightedCombinedLoss(num_classes, weight_ce, weight_dice, class_weights)

    @staticmethod
    def boundary_target(masks, width=1):
        """1 where the label changes within a (2*width+1) window, else 0."""
        m = masks.unsqueeze(1).float()
        k = 2 * width + 1
        local_max = F.max_pool2d(m, k, stride=1, padding=width)
        local_min = -F.max_pool2d(-m, k, stride=1, padding=width)
        return (local_max != local_min).to(m.dtype)

    def forward(self, outputs, targets):
        # eval mode: the model returns segmentation logits only
        if outputs.shape[1] == self.num_classes:
            return self.base(outputs, targets)
        if outputs.shape[1] != self.num_classes + 1:
            raise ValueError(
                f"SOBLoss expected {self.num_classes} or {self.num_classes + 1} output "
                f"channels, got {outputs.shape[1]}.")

        seg = outputs[:, :self.num_classes]
        edge_logits = outputs[:, self.num_classes:self.num_classes + 1]
        loss = self.base(seg, targets)

        target_edge = self.boundary_target(targets.long(), self.edge_width)
        pos = target_edge.sum()
        neg = target_edge.numel() - pos
        pos_weight = torch.clamp(neg / pos.clamp(min=1.0), max=self.pos_weight_cap).view(1)
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits, target_edge, pos_weight=pos_weight)
        return loss + self.weight_edge * edge_loss
