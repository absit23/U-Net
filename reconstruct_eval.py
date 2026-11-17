"""Evaluate a 'crackmerged' model on the ORIGINAL 7-class taxonomy.

The network is trained with Microcrack and Structural crack merged into one
"Crack" class, so it never has to make a decision the image cannot support.
At inference the two classes are recovered by measuring the width of each
predicted crack region and applying a threshold:

    width < tau  -> Microcrack        width >= tau -> Structural crack

The result is a full 8-ID prediction, scored against the untouched ground
truth, so the numbers are directly comparable with evaluate.py on the baseline.

    python reconstruct_eval.py --data_dir data/Try3X_811_ds \
        --checkpoint_path runs/cm/crackmerged_e80/last_unet_mobilenet.pth
"""
import argparse
import os

import cv2
import numpy as np
import torch

from configs import get_class_names
from datasets import build_loader, resolve_split_dirs
from models import build_model
from training.checkpoint import load_checkpoint
from training.metrics import SegmentationMetrics

MICRO, STRUCT, CRACK_MERGED = 3, 7, 3   # IDs: 7class Microcrack / Structural crack, crackmerged Crack


def crack_components(crack_mask, min_area):
    """(component boolean mask, apparent width) for each predicted crack region."""
    n, labels = cv2.connectedComponents(crack_mask.astype(np.uint8), connectivity=8)
    out = []
    for comp in range(1, n):
        sel = labels == comp
        if sel.sum() < min_area:
            continue
        padded = np.pad(sel.astype(np.uint8), 1)
        edt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        width = float(4.0 * np.median(edt[padded > 0]))
        out.append((sel, width))
    return out


@torch.no_grad()
def run(model, loader, taus, min_area, device, small_as):
    """One forward pass; score every threshold in `taus`."""
    metrics = {t: SegmentationMetrics(8) for t in taus}
    inter_c, union_c = 0, 0          # crack-vs-background IoU, severity ignored
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].numpy().astype(np.int64)
        pred_m = model(images).argmax(1).cpu().numpy().astype(np.int64)

        for g, pm in zip(gt, pred_m):
            gt_crack = (g == MICRO) | (g == STRUCT)
            pr_crack = pm == CRACK_MERGED
            inter_c += int((gt_crack & pr_crack).sum())
            union_c += int((gt_crack | pr_crack).sum())

            base = pm.copy()
            base[pr_crack] = 0                       # crack pixels filled in per threshold
            comps = crack_components(pr_crack, min_area)
            leftover = pr_crack.copy()
            for sel, _ in comps:
                leftover &= ~sel                     # regions too small to measure
            for t in taus:
                p8 = base.copy()
                p8[leftover] = small_as              # tiny specks -> narrowest class
                for sel, w in comps:
                    p8[sel] = MICRO if w < t else STRUCT
                metrics[t].update(torch.from_numpy(p8), torch.from_numpy(g))
    crack_iou = inter_c / max(union_c, 1)
    return metrics, crack_iou


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint_path, device)
    merged_names = ckpt.get("class_names") or get_class_names("crackmerged")
    if len(merged_names) != 7 or "Crack" not in merged_names:
        raise ValueError(f"This checkpoint is not a 'crackmerged' model: {merged_names}")
    full_names = get_class_names("7class")
    model_name = ckpt.get("model_name", "unet_mobilenet")
    img_size = args.img_size or ckpt.get("img_size", 640)

    model = build_model(model_name, 3, len(merged_names), pretrained=False,
                        head=ckpt.get("head", "flat"), class_names=merged_names).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    taus = [float(t) for t in args.taus.split(",")]
    print(f"Model: {model_name} | img_size {img_size} | device {device}")
    print(f"Thresholds: {taus}\n")

    all_metrics, crack_ious = {t: SegmentationMetrics(8) for t in taus}, []
    for split in args.splits:
        if resolve_split_dirs(args.data_dir, split) is None:
            print(f"  (skipping '{split}')")
            continue
        # remap=None on purpose: ground truth keeps the original 8 IDs
        loader = build_loader(args.data_dir, split, img_size, args.batch_size,
                              args.num_workers, False)
        print(f"  {split}: {len(loader.dataset)} images")
        m, ciou = run(model, loader, taus, args.min_area, device, args.small_as)
        for t in taus:
            all_metrics[t].confusion_matrix += m[t].confusion_matrix
        crack_ious.append((split, ciou))

    print("\n" + "=" * 74)
    print(" CRACK DETECTION (Microcrack + Structural crack as one region)")
    print("=" * 74)
    for split, ciou in crack_ious:
        print(f"  {split:<8} crack IoU {ciou:.4f}")
    print("  -> compare with the baseline's merged crack IoU (0.82 on valid+test)")

    print("\n" + "=" * 74)
    print(" RECONSTRUCTED 7-CLASS RESULT, BY WIDTH THRESHOLD")
    print("=" * 74)
    print(f"{'tau':>8}{'mIoU':>10}{'F1':>10}{'IoU Micro':>12}{'IoU Struct':>12}")
    best = None
    for t in taus:
        r = all_metrics[t].compute()
        iou = r["per_class_iou"]
        print(f"{t:>8.2f}{r['mIoU']:>10.4f}{r['mDice']:>10.4f}"
              f"{iou[MICRO]:>12.4f}{iou[STRUCT]:>12.4f}")
        if best is None or r["mIoU"] > best[1]["mIoU"]:
            best = (t, r)

    t, r = best
    print("\n" + "=" * 74)
    print(f" PER-CLASS DETAIL AT tau = {t:.2f}")
    print("=" * 74)
    print(f"mIoU {r['mIoU']:.4f} | F1 {r['mDice']:.4f} | "
          f"Precision {r['mPrecision']:.4f} | Recall {r['mRecall']:.4f}\n")
    print(f"{'ID':<5}{'Class':<22}{'IoU':<10}{'Dice':<10}")
    print("-" * 48)
    for i, name in enumerate(full_names):
        print(f"{i:<5}{name:<22}{r['per_class_iou'][i]:<10.4f}{r['per_class_dice'][i]:<10.4f}")
    print("\nNOTE: tau was picked on these same images, so this row is optimistic.")
    print("Choose tau on valid only, then report the test number at that tau.")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "reconstruct_sweep.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("tau,mIoU,F1," + ",".join(n.replace(" ", "_") for n in full_names) + "\n")
        for t in taus:
            r = all_metrics[t].compute()
            f.write(f"{t},{r['mIoU']:.5f},{r['mDice']:.5f},"
                    + ",".join(f"{v:.5f}" for v in r["per_class_iou"]) + "\n")
    print(f"\nSweep table: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--taus", default="8,10,11,12,12.39,13,14,16,18,20")
    p.add_argument("--min_area", type=int, default=30)
    p.add_argument("--small_as", type=int, default=MICRO,
                   help="class for predicted crack specks too small to measure")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", default="runs/analysis")
    main(p.parse_args())
