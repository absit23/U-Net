"""Loss functions used by the recent concrete/structural damage segmentation papers.

The loss is the only factor on the "large gain" list that this project has never
varied: every run so far used weighted CE + Dice. Published ablations attribute
+4 to +7 mIoU points to changing it, which is the one range above this dataset's
measurement floor.

Implemented, with the paper each follows:

  ce_dice       weighted CE + soft Dice                     (current baseline)
  focal_dice    focal + Dice                                 common in this field
  tversky       weighted CE + Tversky (alpha<beta)           rare-class recall
  focal_tversky focal + focal-Tversky                        rare + hard pixels
  celik         0.4 focal + 0.4 Dice + 0.2 skeleton-recall   Celik et al. 2025
  ce_dice_skel  weighted CE + Dice + skeleton-recall         ablation of the above

Tversky trades precision for recall through alpha/beta, which targets the rare
classes (Minor spalling). Skeleton-recall scores the prediction only on the
morphological skeleton of each ground-truth region, so thin structures count as
much as thick ones - aimed at Microcrack.

All of them take logits [B,C,H,W] and integer targets [B,H,W], like
WeightedCombinedLoss, so they are drop-in replacements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import WeightedCombinedLoss

LOSS_CHOICES = ["ce_dice", "ce_dice_s1", "ce_dice_present", "focal_dice", "tversky",
                "focal_tversky", "celik", "ce_dice_skel"]


def _one_hot(targets, num_classes, dtype):
    return F.one_hot(targets, num_classes).permute(0, 3, 1, 2).to(dtype)


@torch.no_grad()
def soft_skeleton(mask, iters=3):
    """Morphological skeleton by iterative erosion/opening (clDice-style).

    The mask comes from the ground truth, so this needs no gradient - which also
    keeps the intermediate tensors off the autograd graph.
    """
    skel = torch.zeros_like(mask)
    img = mask
    for _ in range(iters):
        eroded = -F.max_pool2d(-img, 3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
        skel = torch.clamp(skel + torch.clamp(img - opened, min=0.0), 0.0, 1.0)
        img = eroded
    return skel


class FocalLoss(nn.Module):
    """Multi-class focal loss; class_weights act as the alpha term."""

    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights if class_weights is not None
                             else torch.empty(0))

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.class_weights.numel():
            loss = loss * self.class_weights.to(logits.device)[targets]
        return loss.mean()


class TverskyLoss(nn.Module):
    """Soft Tversky. alpha penalises false positives, beta false negatives.

    beta > alpha favours recall, which is what the rare classes need.
    """

    def __init__(self, num_classes, alpha=0.3, beta=0.7, gamma=1.0, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot(targets, self.num_classes, probs.dtype)
        start = 1 if self.ignore_background else 0
        probs, one_hot = probs[:, start:], one_hot[:, start:]
        dims = (0, 2, 3)
        tp = (probs * one_hot).sum(dims)
        fp = (probs * (1.0 - one_hot)).sum(dims)
        fn = ((1.0 - probs) * one_hot).sum(dims)
        index = (tp + 1e-7) / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        loss = 1.0 - index
        if self.gamma != 1.0:
            loss = loss ** self.gamma
        return loss.mean()


class SkeletonRecallLoss(nn.Module):
    """Recall measured only on the skeleton of each ground-truth region.

    A one-pixel-wide crack contributes as much as a large blob, so thin
    structures stop being drowned out by area-based terms.
    """

    def __init__(self, num_classes, iters=10, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.iters = iters
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        with torch.no_grad():
            one_hot = _one_hot(targets, self.num_classes, probs.dtype)
            skel = soft_skeleton(one_hot, self.iters)
        start = 1 if self.ignore_background else 0
        probs, skel = probs[:, start:], skel[:, start:]
        dims = (0, 2, 3)
        recall = ((probs * skel).sum(dims) + 1e-7) / (skel.sum(dims) + 1e-7)
        present = skel.sum(dims) > 0
        if present.sum() == 0:
            return probs.sum() * 0.0
        return (1.0 - recall[present]).mean()


class CompositeLoss(nn.Module):
    """Weighted sum of named terms, with the channel checks of the base loss."""

    def __init__(self, num_classes, terms):
        super().__init__()
        self.num_classes = num_classes
        self.names = [n for n, _, _ in terms]
        self.weights = [w for _, w, _ in terms]
        self.terms = nn.ModuleList([m for _, _, m in terms])

    def forward(self, logits, targets):
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"Model outputs {logits.shape[1]} classes, "
                             f"loss expects {self.num_classes}.")
        targets = targets.long()
        if targets.numel() and (targets.min() < 0 or targets.max() >= self.num_classes):
            raise ValueError(f"Mask IDs must be in [0, {self.num_classes - 1}], got "
                             f"min={targets.min().item()}, max={targets.max().item()}.")
        total = 0.0
        for w, term in zip(self.weights, self.terms):
            total = total + w * term(logits, targets)
        return total

    def describe(self):
        return " + ".join(f"{w:g}*{n}" for n, w in zip(self.names, self.weights))


class DiceTerm(nn.Module):
    """Soft Dice.

    `smooth` matters far more than it looks. With the usual 1e-7, a class that
    is absent from the batch scores ~0 instead of 1, because the model's tiny
    leftover probabilities dominate a near-zero denominator. A perfect
    prediction on a batch where 4 of 8 classes are absent still returns a Dice
    loss of 0.497, and the gradient is then mostly about suppressing absent
    classes rather than segmenting present ones. Rare classes are absent from
    most batches, so they receive that suppression almost every step.

    smooth=1.0 is the standard fix: absent and correctly-unpredicted classes
    score 1 and contribute nothing. `present_only` instead averages over the
    classes actually in the batch.
    """

    def __init__(self, num_classes, smooth=1e-7, present_only=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.present_only = present_only

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot(targets, self.num_classes, probs.dtype)
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        card = (probs + one_hot).sum(dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        if self.present_only:
            present = one_hot.sum(dims) > 0
            if present.sum() == 0:
                return probs.sum() * 0.0
            dice = dice[present]
        return 1.0 - dice.mean()


class CETerm(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


def build_loss(name, num_classes, class_weights=None, weight_ce=1.0, weight_dice=1.0,
               focal_gamma=2.0, tversky_alpha=0.3, tversky_beta=0.7, ft_gamma=0.75,
               weight_skel=0.2):
    """Return a loss module with the WeightedCombinedLoss interface."""
    name = name.lower()
    if name == "ce_dice":
        return WeightedCombinedLoss(num_classes, weight_ce, weight_dice, class_weights)
    if name == "ce_dice_s1":          # same as ce_dice, but Dice smoothing 1.0
        return CompositeLoss(num_classes, [
            ("ce", weight_ce, CETerm(class_weights)),
            ("dice(s=1)", weight_dice, DiceTerm(num_classes, smooth=1.0)),
        ])
    if name == "ce_dice_present":     # same as ce_dice, but Dice over present classes only
        return CompositeLoss(num_classes, [
            ("ce", weight_ce, CETerm(class_weights)),
            ("dice(present)", weight_dice,
             DiceTerm(num_classes, smooth=1e-7, present_only=True)),
        ])
    if name == "focal_dice":
        return CompositeLoss(num_classes, [
            ("focal", 1.0, FocalLoss(focal_gamma, class_weights)),
            ("dice", 1.0, DiceTerm(num_classes)),
        ])
    if name == "tversky":
        return CompositeLoss(num_classes, [
            ("ce", weight_ce, CETerm(class_weights)),
            ("tversky", 1.0, TverskyLoss(num_classes, tversky_alpha, tversky_beta)),
        ])
    if name == "focal_tversky":
        return CompositeLoss(num_classes, [
            ("focal", 1.0, FocalLoss(focal_gamma, class_weights)),
            ("focal_tversky", 1.0,
             TverskyLoss(num_classes, tversky_alpha, tversky_beta, ft_gamma)),
        ])
    if name == "celik":
        return CompositeLoss(num_classes, [
            ("focal", 0.4, FocalLoss(focal_gamma, class_weights)),
            ("dice", 0.4, DiceTerm(num_classes)),
            ("skeleton_recall", 0.2, SkeletonRecallLoss(num_classes)),
        ])
    if name == "ce_dice_skel":
        return CompositeLoss(num_classes, [
            ("ce", weight_ce, CETerm(class_weights)),
            ("dice", weight_dice, DiceTerm(num_classes)),
            ("skeleton_recall", weight_skel, SkeletonRecallLoss(num_classes)),
        ])
    raise ValueError(f"Unknown --loss '{name}'. Choices: {LOSS_CHOICES}")
