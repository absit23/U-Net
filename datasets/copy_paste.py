"""Copy-paste augmentation for rare classes (train split only).

For a training image, with probability `prob`, take 1..`max_pastes` connected regions
of a rare class from OTHER training images and paste them (pixels + mask labels) at a
random position. The donors always come from the same (train) dataset, so nothing leaks
from valid / test. The usual flips / rotations / colour changes are applied afterwards.
"""
import os
import random

import cv2
import numpy as np


class CopyPaste:
    def __init__(self, dataset, classes, prob=0.5, max_pastes=2, min_area=30):
        self.ds = dataset
        self.classes = list(classes)
        self.prob = prob
        self.max_pastes = max_pastes
        self.min_area = min_area
        # index: class id -> list of donor image indices that contain it
        self.donors = {c: [] for c in self.classes}
        for i, name in enumerate(dataset.image_filenames):
            m = cv2.imread(dataset._find_mask_path(name), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            present = np.unique(m)
            for c in self.classes:
                if c in present and int((m == c).sum()) >= min_area:
                    self.donors[c].append(i)
        self.classes = [c for c in self.classes if self.donors[c]]

    def summary(self):
        return {c: len(v) for c, v in self.donors.items()}

    def _load(self, i):
        name = self.ds.image_filenames[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.ds.images_dir, name)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.ds._find_mask_path(name), cv2.IMREAD_GRAYSCALE)
        return img, msk

    def __call__(self, image, mask, self_idx=None):
        if not self.classes or random.random() >= self.prob:
            return image, mask
        image, mask = image.copy(), mask.copy()
        for _ in range(random.randint(1, self.max_pastes)):
            c = random.choice(self.classes)          # classes are sampled uniformly
            cands = [i for i in self.donors[c] if i != self_idx]
            if not cands:
                continue
            d_img, d_msk = self._load(random.choice(cands))
            n, lab, stats, _ = cv2.connectedComponentsWithStats((d_msk == c).astype(np.uint8), connectivity=8)
            comps = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= self.min_area]
            if not comps:
                continue
            k = random.choice(comps)
            x, y, w, h = (int(stats[k, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                     cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            patch = d_img[y:y + h, x:x + w].copy()
            region = (lab[y:y + h, x:x + w] == k)
            if random.random() < 0.5:                 # random flips of the patch
                patch, region = patch[:, ::-1], region[:, ::-1]
            if random.random() < 0.5:
                patch, region = patch[::-1], region[::-1]
            ph, pw = region.shape
            H, W = mask.shape
            if ph > H or pw > W:
                continue
            ty, tx = random.randint(0, H - ph), random.randint(0, W - pw)
            sub_img = image[ty:ty + ph, tx:tx + pw]
            sub_msk = mask[ty:ty + ph, tx:tx + pw]
            sub_img[region] = patch[region]
            sub_msk[region] = c
        return image, mask
