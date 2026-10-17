import numpy as np
import torch


class SegmentationMetrics:
    """Confusion-matrix based metrics. Rows = ground truth, columns = prediction."""

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)
        t = targets.detach().cpu().numpy().flatten()
        p = preds.detach().cpu().numpy().flatten()
        keep = (t >= 0) & (t < self.num_classes) & (t != self.ignore_index)
        idx = self.num_classes * t[keep].astype(int) + p[keep].astype(int)
        self.confusion_matrix += np.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    @staticmethod
    def _ratio(num, den):
        """num/den, NaN where den == 0 (class absent) so it is skipped by nanmean."""
        out = np.full(num.shape, np.nan)
        np.divide(num, den, out=out, where=den > 0)
        return out

    def compute(self, include_background=False):
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou = self._ratio(tp, tp + fp + fn)
        dice = self._ratio(2 * tp, 2 * tp + fp + fn)
        precision = self._ratio(tp, tp + fp)
        recall = self._ratio(tp, tp + fn)

        s = 0 if include_background else 1  # macro averages skip background (ID 0)
        return {
            "mIoU": float(np.nanmean(iou[s:])),
            "mDice": float(np.nanmean(dice[s:])),
            "mPrecision": float(np.nanmean(precision[s:])),
            "mRecall": float(np.nanmean(recall[s:])),
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "confusion_matrix": cm.tolist(),
        }
