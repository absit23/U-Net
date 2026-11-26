"""Check the dacl10k export and report what the loader actually sees.

Run this before any training. It confirms the folders, verifies the class list
against meta.json, and reports per-class image counts and pixel share, so the
numbers can be cross-checked against the dataset's published statistics.

    python dacl10k_inspect.py --root path/to/dacl10k --sample 400
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

from datasets.dacl10k import (CLASS_TO_IDX, CONCRETE_DEFECTS, DACL_CLASSES,
                              GENERAL_DEFECTS, NUM_CLASSES, OBJECT_PARTS,
                              Dacl10kDataset, polygons_to_mask, verify_meta)

EXPECTED = {"train": 6935, "val": 975, "test": 2010}


def group_of(name):
    if name in CONCRETE_DEFECTS:
        return "concrete"
    if name in GENERAL_DEFECTS:
        return "general"
    return "object"


def main(args):
    print("=" * 74)
    print(" FOLDERS")
    print("=" * 74)
    splits = []
    for split in ("train", "val", "test"):
        img_dir = os.path.join(args.root, split, "img")
        ann_dir = os.path.join(args.root, split, "ann")
        if not os.path.isdir(img_dir):
            print(f"  {split:<6} MISSING")
            continue
        n_img = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        n_ann = len(os.listdir(ann_dir)) if os.path.isdir(ann_dir) else 0
        flag = "" if EXPECTED.get(split) in (None, n_img) else f"  <-- expected {EXPECTED[split]}"
        print(f"  {split:<6} images {n_img:>6}   annotations {n_ann:>6}{flag}")
        splits.append(split)

    print("\n" + "=" * 74)
    print(" CLASS LIST")
    print("=" * 74)
    try:
        verify_meta(args.root)
        print(f"  meta.json matches the {NUM_CLASSES} classes in the code.")
    except ValueError as exc:
        print(f"  MISMATCH\n{exc}")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read meta.json: {exc}")

    # ---- per-class statistics over a sample of the training split -----------
    split = args.split
    ds = Dacl10kDataset(args.root, split, img_size=args.img_size, transform=None,
                        strict=(split != "test"))
    names = ds.files[:args.sample] if args.sample else ds.files
    print(f"\n  scanning {len(names)} of {len(ds.files)} '{split}' annotations ...")

    img_count = Counter()          # images containing the class
    pixel_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    pair_count = Counter()         # images containing both classes
    overlap_px = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    unknown_total = 0
    empty_images = 0

    for name in names:
        path = ds.ann_path(name)
        if path is None:
            continue
        with open(path, encoding="utf-8") as f:
            ann = json.load(f)
        mask, unknown = polygons_to_mask(ann, args.img_size, args.img_size)
        unknown_total += unknown
        present = np.where(mask.any(axis=(0, 1)))[0]
        if present.size == 0:
            empty_images += 1
            continue
        pixel_sum += mask.reshape(-1, NUM_CLASSES).mean(axis=0)
        for i in present:
            img_count[int(i)] += 1
        for a_i, a in enumerate(present):
            for b in present[a_i + 1:]:
                pair_count[(int(a), int(b))] += 1
                both = np.logical_and(mask[:, :, a], mask[:, :, b]).mean()
                overlap_px[a, b] += both

    n = max(len(names) - empty_images, 1)
    print(f"  images with no annotated object: {empty_images}")
    if unknown_total:
        print(f"  WARNING: {unknown_total} objects had a class outside the 19 - check meta.json")

    print("\n" + "=" * 74)
    print(f" PER-CLASS over '{split}' ({len(names)} images, rasterised at {args.img_size}px)")
    print("=" * 74)
    print(f"{'class':<30}{'group':<10}{'images':>8}{'% of imgs':>11}{'mean px share':>15}")
    print("-" * 74)
    order = np.argsort(-np.array([img_count.get(i, 0) for i in range(NUM_CLASSES)]))
    for i in order:
        name = DACL_CLASSES[i]
        c = img_count.get(int(i), 0)
        print(f"{name:<30}{group_of(name):<10}{c:>8}{100.0 * c / len(names):>10.1f}%"
              f"{100.0 * pixel_sum[i] / n:>14.3f}%")

    # ---- the pairs that actually share pixels -------------------------------
    print("\n" + "=" * 74)
    print(" TOP CO-OCCURRING PAIRS (same image)")
    print("=" * 74)
    for (a, b), c in pair_count.most_common(12):
        mean_overlap = 100.0 * overlap_px[a, b] / max(c, 1)
        print(f"  {DACL_CLASSES[a]:<28} + {DACL_CLASSES[b]:<28} {c:>5} imgs"
              f"   shared pixels {mean_overlap:>6.2f}%")

    print("\nNOTE: dacl10k is multi-label - a pixel may carry several classes, so the")
    print("per-class shares do not sum to 100%. Train with a sigmoid per channel.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder holding train/ val/ test/ and meta.json")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--sample", type=int, default=400, help="0 = scan the whole split")
    p.add_argument("--img_size", type=int, default=512)
    main(p.parse_args())
