import os

import cv2
import numpy as np

from configs import default_class_weights
from datasets import resolve_split_dirs, ConcreteDamageDataset


def compute_pixel_frequencies(data_dir, num_classes, split="train"):
    """Count pixels per class ID over the masks of a split."""
    dirs = resolve_split_dirs(data_dir, split)
    ds = ConcreteDamageDataset(*dirs)
    counts = np.zeros(num_classes, dtype=np.int64)
    for name in ds.image_filenames:
        mask = cv2.imread(ds._find_mask_path(name), cv2.IMREAD_GRAYSCALE)
        counts += np.bincount(mask.flatten(), minlength=num_classes)[:num_classes]
    return counts


def auto_weights(counts, min_w=0.1, max_w=20.0):
    """Median-frequency balancing, clipped. Classes with no pixels get weight 1."""
    freq = counts / max(counts.sum(), 1)
    present = freq > 0
    weights = np.ones(len(counts))
    weights[present] = np.median(freq[present]) / freq[present]
    return np.clip(weights, min_w, max_w).tolist()


def resolve_class_weights(arg, class_names, data_dir):
    """arg: None -> manual defaults, 'auto' -> from train masks, 'a,b,c' -> explicit list."""
    n = len(class_names)
    if arg is None:
        return default_class_weights(class_names)
    if arg == "auto":
        counts = compute_pixel_frequencies(data_dir, n)
        weights = auto_weights(counts)
        print("Pixel counts per class:", dict(zip(class_names, counts.tolist())))
        print("Auto class weights:   ", dict(zip(class_names, [round(w, 3) for w in weights])))
        return weights
    weights = [float(v) for v in arg.split(",")]
    if len(weights) != n:
        raise ValueError(f"Got {len(weights)} class weights for {n} classes (order: {class_names}).")
    return weights
