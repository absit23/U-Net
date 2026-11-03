"""Build a copy of the dataset where Delamination + Rebar corrosion are merged
into one class ("Severe spalling"). The source dataset is not modified.

    python make_merged_dataset.py --src data\\dataset_v1 --dst data\\dataset_v1_merged

Then train with --variant merged6 --data_dir data\\dataset_v1_merged.
"""
import argparse
import os
import shutil

import cv2
import numpy as np

# old IDs: 0 Bg, 1 Delam, 2 Effl, 3 Micro, 4 Minor, 5 Moderate, 6 Rebar, 7 Struct
LUT = np.array([0, 5, 1, 2, 3, 4, 5, 6], dtype=np.uint8)
OLD = ["Background", "Delamination", "Efflorescence", "Microcrack",
       "Minor spalling", "Moderate spalling", "Rebar corrosion", "Structural crack"]
NEW = ["Background", "Efflorescence", "Microcrack", "Minor spalling",
       "Moderate spalling", "Severe spalling", "Structural crack"]


def is_mask(path):
    low = os.path.basename(path).lower()
    return low.endswith("_mask.png") or os.path.basename(os.path.dirname(path)).lower() == "masks"


def sqrt_weights(counts):
    freq = counts / counts.sum()
    return np.sqrt(np.median(freq) / freq)


def main(src, dst):
    old_counts = np.zeros(8, dtype=np.int64)
    new_counts = np.zeros(7, dtype=np.int64)
    n_masks = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        is_train = rel.replace("\\", "/").split("/")[0] == "train"
        for f in files:
            sp, dp = os.path.join(root, f), os.path.join(dst, rel, f)
            if not is_mask(sp):
                shutil.copy2(sp, dp)
                continue
            m = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
            assert m.max() <= 7, f"unexpected mask value in {sp}"
            nm = LUT[m]
            cv2.imwrite(dp, nm)
            n_masks += 1
            if is_train:
                old_counts += np.bincount(m.ravel(), minlength=8)
                new_counts += np.bincount(nm.ravel(), minlength=7)

    print(f"\nRemapped {n_masks} masks -> {dst}")
    print("\nOLD train pixel share (%) and sqrt-median weights (check vs your w_sqrt):")
    for n, c, w in zip(OLD, old_counts, sqrt_weights(old_counts)):
        print(f"  {n:20s} {100 * c / old_counts.sum():6.2f}   {w:.3f}")
    print("  your w_sqrt: 0.316,0.854,1.515,2.716,4.472,1.245,0.436,0.859")
    print("\nNEW train pixel share (%) and sqrt-median weights:")
    nw = sqrt_weights(new_counts)
    for n, c, w in zip(NEW, new_counts, nw):
        print(f"  {n:20s} {100 * c / new_counts.sum():6.2f}   {w:.3f}")
    print("\n--class_weights " + ",".join(f"{w:.3f}" for w in nw))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    a = p.parse_args()
    main(a.src, a.dst)
