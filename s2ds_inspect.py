"""Inspect the S2DS dataset: folder layout, image sizes, and the colours used in
the label images, so the colour -> class mapping can be written down exactly.

    python s2ds_inspect.py --root path\\to\\s2ds
"""
import argparse
import os
from collections import Counter

import cv2
import numpy as np


def walk(root, max_depth=3):
    print("=" * 70)
    print(" FOLDER LAYOUT")
    print("=" * 70)
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        imgs = [f for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        labs = [f for f in imgs if "_lab" in f.lower()]
        indent = "  " * depth
        name = os.path.basename(dirpath) or dirpath
        extra = f"   [{len(imgs) - len(labs)} images, {len(labs)} labels]" if imgs else ""
        print(f"{indent}{name}/{extra}")
        if imgs:
            print(f"{indent}   e.g. {sorted(imgs)[:4]}")


def find_pairs(root, limit=None):
    """(image_path, label_path) pairs, matching 000.png with 000_lab.png."""
    pairs = []
    for dirpath, _, filenames in os.walk(root):
        labels = {f for f in filenames if "_lab" in f.lower()}
        for lab in sorted(labels):
            stem = lab.lower().replace("_lab", "")
            for cand in filenames:
                if cand.lower() == stem and "_lab" not in cand.lower():
                    pairs.append((os.path.join(dirpath, cand), os.path.join(dirpath, lab)))
                    break
        if limit and len(pairs) >= limit:
            break
    return pairs


def main(args):
    walk(args.root)
    pairs = find_pairs(args.root)
    print(f"\nMatched image/label pairs: {len(pairs)}")
    if not pairs:
        print("No pairs found - send me the folder listing and I'll adjust the matching.")
        return

    print("\n" + "=" * 70)
    print(" IMAGE SIZES (first 5 pairs)")
    print("=" * 70)
    for img_p, lab_p in pairs[:5]:
        im = cv2.imread(img_p)
        lb = cv2.imread(lab_p)
        print(f"  {os.path.basename(img_p):<16} image {im.shape if im is not None else 'unreadable'}"
              f"   label {lb.shape if lb is not None else 'unreadable'}")

    print("\n" + "=" * 70)
    print(f" LABEL COLOURS over {min(args.n, len(pairs))} label images")
    print("=" * 70)
    counter = Counter()
    for _, lab_p in pairs[:args.n]:
        lab = cv2.imread(lab_p, cv2.IMREAD_COLOR)
        if lab is None:
            continue
        flat = lab.reshape(-1, 3)
        colours, counts = np.unique(flat, axis=0, return_counts=True)
        for colour, count in zip(colours, counts):
            b, g, r = (int(v) for v in colour)
            counter[(r, g, b)] += int(count)

    total = sum(counter.values()) or 1
    print(f"{'R':>5}{'G':>5}{'B':>5}{'pixels':>14}{'share':>9}")
    print("-" * 40)
    for (r, g, b), count in counter.most_common(args.top):
        print(f"{r:>5}{g:>5}{b:>5}{count:>14,}{100.0 * count / total:>8.2f}%")
    print(f"\nDistinct colours found: {len(counter)}")
    if len(counter) > 20:
        print("NOTE: many colours usually means the labels were saved with JPEG\n"
              "compression or anti-aliasing. Tell me and I'll snap to nearest class.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder containing the S2DS data")
    p.add_argument("--n", type=int, default=15, help="how many label images to scan")
    p.add_argument("--top", type=int, default=25, help="how many colours to list")
    main(p.parse_args())
