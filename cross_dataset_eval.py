"""Zero-shot cross-dataset evaluation: a model trained on Try3X_811_ds, tested on S2DS.

No training and no fine-tuning happens here. The model has never seen a single
S2DS image, so this measures how far the model carries beyond the distribution
it was fitted on - the objection a reviewer raises first against a single-dataset
study.

Taxonomy mapping (7class -> S2DS):

    Background                      -> background
    Microcrack, Structural crack    -> crack
    Minor spalling, Moderate spall. -> spalling
    Rebar corrosion                 -> corrosion
    Efflorescence                   -> efflorescence
    Delamination                    -> spalling, or ignored (--delamination)

S2DS vegetation and control point have no counterpart in the 7class taxonomy;
the model was never trained to predict them, so those ground-truth pixels are
excluded rather than counted as errors. State this in the paper.

S2DS label colours, taken from the dataset's own colour swatches:
    background (0,0,0)  crack (255,255,255)  spalling (255,0,0)
    corrosion (255,255,0)  efflorescence (0,255,255)
    vegetation (0,255,0)  control point (0,0,255)

    python cross_dataset_eval.py --s2ds_root path/to/s2ds \
        --checkpoint_path runs/BASELINE_v1/last_unet_mobilenet.pth
"""
import argparse
import os

import cv2
import numpy as np
import torch

from configs import get_class_names
from models import build_model
from training.checkpoint import load_checkpoint

IGNORE = 255

# S2DS colour (R,G,B) -> name
S2DS_COLOURS = {
    (0, 0, 0): "background",
    (255, 255, 255): "crack",
    (255, 0, 0): "spalling",
    (255, 255, 0): "corrosion",
    (0, 255, 255): "efflorescence",
    (0, 255, 0): "vegetation",
    (0, 0, 255): "control_point",
}

# the classes we score, in order
EVAL_CLASSES = ["background", "crack", "spalling", "corrosion", "efflorescence"]
EVAL_INDEX = {name: i for i, name in enumerate(EVAL_CLASSES)}
# present in S2DS but outside the trained taxonomy
EXCLUDED = ["vegetation", "control_point"]

# 7class name -> S2DS name
SEVEN_TO_S2DS = {
    "Background": "background",
    "Microcrack": "crack",
    "Structural crack": "crack",
    "Minor spalling": "spalling",
    "Moderate spalling": "spalling",
    "Rebar corrosion": "corrosion",
    "Efflorescence": "efflorescence",
    # Delamination handled by --delamination
}


def colour_key(rgb):
    r, g, b = rgb
    return (int(r) << 16) | (int(g) << 8) | int(b)


def build_colour_lut():
    """colour key -> eval index (or IGNORE)."""
    lut = {}
    for rgb, name in S2DS_COLOURS.items():
        lut[colour_key(rgb)] = IGNORE if name in EXCLUDED else EVAL_INDEX[name]
    return lut


def label_to_eval(label_bgr, lut):
    """Colour label image -> eval indices. Unknown colours become IGNORE."""
    b, g, r = label_bgr[:, :, 0], label_bgr[:, :, 1], label_bgr[:, :, 2]
    keys = (r.astype(np.int32) << 16) | (g.astype(np.int32) << 8) | b.astype(np.int32)
    out = np.full(keys.shape, IGNORE, dtype=np.uint8)
    unknown = np.ones(keys.shape, dtype=bool)
    for key, idx in lut.items():
        hit = keys == key
        out[hit] = idx
        unknown &= ~hit
    return out, int(unknown.sum())


def build_pred_map(class_names, delamination):
    """7class ID -> eval index (or IGNORE)."""
    mapping = np.full(len(class_names), IGNORE, dtype=np.uint8)
    for i, name in enumerate(class_names):
        if name in SEVEN_TO_S2DS:
            mapping[i] = EVAL_INDEX[SEVEN_TO_S2DS[name]]
        elif name == "Delamination":
            mapping[i] = IGNORE if delamination == "ignore" else EVAL_INDEX["spalling"]
    return mapping


def find_pairs(root, splits):
    pairs = []
    for split in splits:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            print(f"  (no '{split}' folder)")
            continue
        files = os.listdir(d)
        labels = sorted(f for f in files if "_lab" in f.lower())
        n = 0
        for lab in labels:
            stem = lab.lower().replace("_lab", "")
            for cand in files:
                if cand.lower() == stem and "_lab" not in cand.lower():
                    pairs.append((split, os.path.join(d, cand), os.path.join(d, lab)))
                    n += 1
                    break
        print(f"  {split}: {n} pairs")
    return pairs


def confusion(pred, gt, n):
    valid = (gt != IGNORE) & (pred != IGNORE)
    if not valid.any():
        return np.zeros((n, n), dtype=np.int64)
    idx = gt[valid].astype(np.int64) * n + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def iou_per_class(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = tp + fp + fn
    return np.divide(tp, den, out=np.full(len(tp), np.nan), where=den > 0)


def bootstrap_ci(cms, n_classes, resamples, rng):
    """95% CI of the defect mIoU, resampling images."""
    stack = np.stack(cms)
    vals = []
    for _ in range(resamples):
        pick = rng.integers(0, len(stack), len(stack))
        iou = iou_per_class(stack[pick].sum(0))[1:]
        vals.append(np.nanmean(iou))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint_path, device)
    class_names = ckpt.get("class_names") or get_class_names("7class")
    img_size = args.img_size or ckpt.get("img_size", 640)

    model = build_model(ckpt.get("model_name", "unet_mobilenet"), 3, len(class_names),
                        pretrained=False, head=ckpt.get("head", "flat"),
                        class_names=class_names,
                        use_bag=ckpt.get("use_bag", False),
                        use_osh=ckpt.get("use_osh", False),
                        n_stages=ckpt.get("n_stages", 4)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pred_map = build_pred_map(class_names, args.delamination)
    lut = build_colour_lut()
    n = len(EVAL_CLASSES)

    print(f"Model: {ckpt.get('model_name')} | img_size {img_size} | device {device}")
    print(f"Delamination -> {'ignored' if args.delamination == 'ignore' else 'spalling'}")
    print("Mapping: " + ", ".join(
        f"{cn}->{EVAL_CLASSES[pred_map[i]] if pred_map[i] != IGNORE else 'ignored'}"
        for i, cn in enumerate(class_names)))
    print(f"\nScanning {args.s2ds_root} ...")
    pairs = find_pairs(args.s2ds_root, args.splits)
    if not pairs:
        print("No image/label pairs found.")
        return

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    per_split, all_cms, unknown_total = {}, [], 0

    for split, img_p, lab_p in pairs:
        image = cv2.imread(img_p, cv2.IMREAD_COLOR)
        label = cv2.imread(lab_p, cv2.IMREAD_COLOR)
        if image is None or label is None:
            continue
        gt, unknown = label_to_eval(label, lut)
        unknown_total += unknown

        resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)) \
                      .unsqueeze(0).to(device)
        logits = model(tensor)
        pred_small = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred_full = cv2.resize(pred_small, (gt.shape[1], gt.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        pred = pred_map[pred_full]

        cm = confusion(pred, gt, n)
        all_cms.append(cm)
        per_split.setdefault(split, []).append(cm)

    if unknown_total:
        print(f"\nWARNING: {unknown_total:,} label pixels had colours outside the "
              f"documented palette and were ignored.")

    print("\n" + "=" * 70)
    print(" ZERO-SHOT CROSS-DATASET RESULT (trained on Try3X_811_ds, tested on S2DS)")
    print("=" * 70)
    for split, cms in per_split.items():
        iou = iou_per_class(np.stack(cms).sum(0))
        print(f"  {split:<8} n={len(cms):<4} defect mIoU {np.nanmean(iou[1:]):.4f}")

    total = np.stack(all_cms).sum(0)
    iou = iou_per_class(total)
    rng = np.random.default_rng(0)
    lo, hi = bootstrap_ci(all_cms, n, args.resamples, rng)
    miou = float(np.nanmean(iou[1:]))

    print(f"\n  ALL {len(all_cms)} images: defect mIoU {miou:.4f} "
          f"  95% CI [{lo:.4f}, {hi:.4f}]   = {miou:.4f} +/- {(hi - lo) / 2:.4f}")
    print("\n" + f"{'Class':<16}{'IoU':>10}{'GT pixels':>16}")
    print("-" * 44)
    for i, name in enumerate(EVAL_CLASSES):
        print(f"{name:<16}{iou[i]:>10.4f}{total[i].sum():>16,}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(os.path.dirname(args.checkpoint_path))
    out = os.path.join(args.out_dir, f"s2ds_{tag}.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("class,iou,gt_pixels\n")
        for i, name in enumerate(EVAL_CLASSES):
            f.write(f"{name},{iou[i]:.5f},{total[i].sum()}\n")
        f.write(f"defect_mIoU,{miou:.5f},\nci_low,{lo:.5f},\nci_high,{hi:.5f},\n")
    print(f"\nSaved: {out}")
    print("\nNOTE: S2DS vegetation and control-point pixels are excluded - the model "
          "was never trained on them. Report this alongside the number.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--s2ds_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--delamination", default="spalling", choices=["spalling", "ignore"])
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--out_dir", default="runs/analysis")
    main(p.parse_args())
