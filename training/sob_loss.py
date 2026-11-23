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
                 weight_edge=0.3, edge_width=1, pos_weight_cap=20.0,
                 ordinal_chain=None, weight_ordinal=0.3, base=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight_edge = weight_edge
        self.edge_width = int(edge_width)
        self.pos_weight_cap = pos_weight_cap
        self.ordinal_chain = list(ordinal_chain or [])
        self.weight_ordinal = weight_ordinal
        self.n_edge = 0          # set by the caller to 1 when BAG is on
        self.n_ordinal = 0       # set by the caller to len(chain)-1 when OSH is on
        self.base = base if base is not None else WeightedCombinedLoss(
            num_classes, weight_ce, weight_dice, class_weights)

    def enable(self, use_bag, use_osh):
        """Declare which auxiliary heads the model emits, so channels line up."""
        self.n_edge = 1 if use_bag else 0
        self.n_ordinal = (len(self.ordinal_chain) - 1) if (use_osh and self.ordinal_chain) else 0
        return self

    def ordinal_loss(self, logits, targets):
        """Cumulative (CORAL) BCE over the damage stages, on chain pixels only."""
        stage = torch.full_like(targets, -1, dtype=torch.long)
        for k, class_id in enumerate(self.ordinal_chain):
            stage[targets == class_id] = k
        valid = (stage >= 0).unsqueeze(1)                      # [B,1,H,W]
        if valid.sum() == 0:
            return logits.sum() * 0.0                          # keeps the graph alive
        levels = torch.arange(1, len(self.ordinal_chain), device=logits.device)
        target = (stage.unsqueeze(1) >= levels.view(1, -1, 1, 1)).to(logits.dtype)
        per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (per_pixel * valid).sum() / (valid.sum() * logits.shape[1]).clamp(min=1)

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

        expected = self.num_classes + self.n_edge + self.n_ordinal
        if outputs.shape[1] != expected:
            raise ValueError(
                f"SOBLoss expected {self.num_classes} or {expected} output channels, "
                f"got {outputs.shape[1]}.")

        seg = outputs[:, :self.num_classes]
        loss = self.base(seg, targets)
        targets_long = targets.long()
        cursor = self.num_classes

        if self.n_edge:
            edge_logits = outputs[:, cursor:cursor + 1]
            cursor += 1
            target_edge = self.boundary_target(targets_long, self.edge_width)
            pos = target_edge.sum()
            neg = target_edge.numel() - pos
            pos_weight = torch.clamp(neg / pos.clamp(min=1.0), max=self.pos_weight_cap).view(1)
            loss = loss + self.weight_edge * F.binary_cross_entropy_with_logits(
                edge_logits, target_edge, pos_weight=pos_weight)

        if self.n_ordinal:
            ord_logits = outputs[:, cursor:cursor + self.n_ordinal]
            loss = loss + self.weight_ordinal * self.ordinal_loss(ord_logits, targets_long)

        return loss
