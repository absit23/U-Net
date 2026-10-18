"""Convert a Roboflow 'COCO Segmentation' export into index-mask folders.

Input : <src>/{train,valid,test}/*.jpg + _annotations.coco.json
Output: <out>/{train,valid,test}/images/*.jpg + masks/*.png  (pixel value = class ID)

Class IDs come from configs/classes.py and are matched by NAME (not by Roboflow's
category number), so the IDs always follow the order defined there.

    python -m datasets.coco_to_masks --src /content/dataset71-1 --out /content/dataset --variant 7class
"""
import argparse
import json
import os
import re
import shutil

import cv2
import numpy as np

from configs import get_class_names

ANN_FILE = "_annotations.coco.json"


def _norm(name):
    return re.sub(r"[\s_\-]+", " ", name.strip().lower())


def convert_split(src_split, out_split, name_to_id, ignore):
    with open(os.path.join(src_split, ANN_FILE), "r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    used_cat_ids = {a["category_id"] for a in coco["annotations"]}

    cat_to_class = {}
    for cid in used_cat_ids:
        key = _norm(cats[cid])
        if key in ignore:
            cat_to_class[cid] = 0
        elif key in name_to_id:
            cat_to_class[cid] = name_to_id[key]
        else:
            raise ValueError(
                f"COCO category '{cats[cid]}' is not in configs/classes.py and not ignored. "
                f"Known: {sorted(name_to_id)}")

    img_out = os.path.join(out_split, "images")
    mask_out = os.path.join(out_split, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    counts = np.zeros(len(name_to_id) + 1, dtype=np.int64)  # pixels per class ID
    skipped_rle = 0
    for im in coco["images"]:
        path = os.path.join(src_split, im["file_name"])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        sx, sy = w / im["width"], h / im["height"]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Paint large regions first so small defects stay visible on top.
        for a in sorted(anns_by_img.get(im["id"], []), key=lambda x: -x.get("area", 0)):
            seg = a["segmentation"]
            if not isinstance(seg, list):  # RLE not supported
                skipped_rle += 1
                continue
            cls = cat_to_class[a["category_id"]]
            for poly in seg:
                pts = np.array(poly, dtype=np.float64).reshape(-1, 2) * [sx, sy]
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], int(cls))

        stem = os.path.splitext(im["file_name"])[0]
        shutil.copy(path, os.path.join(img_out, im["file_name"]))
        cv2.imwrite(os.path.join(mask_out, stem + ".png"), mask)
        counts += np.bincount(mask.flatten(), minlength=len(counts))[:len(counts)]
    return len(coco["images"]), counts, skipped_rle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Roboflow COCO export folder")
    p.add_argument("--out", required=True, help="Output folder (train/valid/test with images+masks)")
    p.add_argument("--variant", default="7class", choices=["6class", "7class"])
    p.add_argument("--ignore", nargs="*", default=[], help="Class names to paint as background")
    args = p.parse_args()

    class_names = get_class_names(args.variant)
    name_to_id = {_norm(n): i for i, n in enumerate(class_names) if i > 0}
    ignore = {_norm(n) for n in args.ignore}

    for split in ("train", "valid", "test"):
        src_split = os.path.join(args.src, split)
        if not os.path.isfile(os.path.join(src_split, ANN_FILE)):
            print(f"[{split}] not found, skipping")
            continue
        n, counts, rle = convert_split(src_split, os.path.join(args.out, split), name_to_id, ignore)
        print(f"[{split}] {n} images converted" + (f" ({rle} RLE annotations skipped)" if rle else ""))
        for i, name in enumerate(class_names):
            print(f"    {i} {name:<20} {counts[i]:>12,} px")
    print(f"\nDone. Masks use IDs from variant '{args.variant}' ({len(class_names)} IDs incl. background).")


if __name__ == "__main__":
    main()
