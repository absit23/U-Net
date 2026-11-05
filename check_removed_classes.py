"""Find images whose Concrete crushing / Structural deformation annotations were
silently turned into background when v1 was built from the original dataset.

    python check_removed_classes.py --v1 data\\dataset_v1 --orig data\\dataset_orig

--orig is the folder made by datasets.coco_to_masks with --variant orig8
(<orig>/<split>/images + masks). --v1 is the Roboflow flat export (jpg + _mask.png).
Images are matched by the name before '_jpg.rf.'.
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np

REMOVED = (1, 8)  # orig8 IDs: 1 Concrete crushing, 8 Structural deformation


def base(name):
    return re.split(r"_jpg\.rf\.|\.rf\.", os.path.splitext(name)[0])[0]


def main(v1, orig, top):
    orig_masks, dup = {}, 0
    for split in ("train", "valid", "test"):
        for p in glob.glob(os.path.join(orig, split, "masks", "*.png")):
            b = base(os.path.basename(p))
            dup += b in orig_masks
            orig_masks[b] = (split, p)
    print(f"original masks: {len(orig_masks)} (duplicate base names: {dup})")

    matched, rows, v1_names = 0, [], set()
    for split in ("train", "valid", "test"):
        for p in glob.glob(os.path.join(v1, split, "*_mask.png")):
            b = base(os.path.basename(p))
            v1_names.add(b)
            if b not in orig_masks:
                continue
            matched += 1
            om = cv2.imread(orig_masks[b][1], cv2.IMREAD_GRAYSCALE)
            vm = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if om is None or vm is None or om.shape != vm.shape:
                continue
            gone = np.isin(om, REMOVED)
            n = int(gone.sum())
            if n:
                rows.append((n, int((vm[gone] == 0).sum()), split, os.path.basename(p), orig_masks[b][0]))
    only_v1 = len(v1_names) - matched
    print(f"v1 images: {len(v1_names)} | matched to the original: {matched} | only in v1 (e.g. added Efflorescence images): {only_v1}")

    deleted = [b for b in orig_masks if b not in v1_names]
    with_removed = 0
    for b in deleted:
        om = cv2.imread(orig_masks[b][1], cv2.IMREAD_GRAYSCALE)
        with_removed += int(np.isin(om, REMOVED).any())
    print(f"original images NOT in v1: {len(deleted)}, of which contain crushing/deformation: {with_removed}")

    rows.sort(reverse=True)
    tot, bg = sum(r[0] for r in rows), sum(r[1] for r in rows)
    print(f"\nimages kept in v1 that had crushing/deformation in the original: {len(rows)}")
    if rows:
        print(f"their removed pixels: {tot:,}; of these, background in v1: {bg:,} ({100 * bg / tot:.1f}%)")
        by = {}
        for r in rows:
            by[r[2]] = by.get(r[2], 0) + 1
        print("per v1 split:", by)
        print(f"\ntop {top} images by removed pixels (px, px that are background in v1, v1 split, file):")
        for n, b0, sp, name, osplit in rows[:top]:
            print(f"  {n:>9,} {b0:>9,}  {sp:<6} {name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--v1", required=True)
    p.add_argument("--orig", required=True)
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()
    main(a.v1, a.orig, a.top)
