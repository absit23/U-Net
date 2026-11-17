"""Does the model's error depend on how wide the crack is?

For every ground-truth instance (connected component) of the chosen classes we
measure its apparent width and ask what the model called it (majority predicted
label over that instance's pixels). Accuracy is then reported per width bin.

Reading the result:
  A) accuracy collapses only where the two classes overlap in width
     -> the class boundary itself is the problem (reformulate the task)
  B) accuracy is poor at every width, including clearly wide instances
     -> the model is not using the width cue (architecture / resolution)

Usage:
    python boundary_error_analysis.py --data_dir data/Try3X_811_ds \
        --checkpoint_path runs/BASELINE_v1/last_unet_mobilenet.pth
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


def instance_records(gt, pred, class_ids, min_area):
    """For one image: (class_id, width_px, area, predicted_class) per GT instance."""
    out = []
    for cid in class_ids:
        binary = (gt == cid).astype(np.uint8)
        if binary.sum() == 0:
            continue
        n, labels = cv2.connectedComponents(binary, connectivity=8)
        for comp in range(1, n):
            sel = labels == comp
            area = int(sel.sum())
            if area < min_area:
                continue
            comp_mask = np.pad(sel.astype(np.uint8), 1)
            edt = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            width = float(4.0 * np.median(edt[comp_mask > 0]))
            pred_label = int(np.bincount(pred[sel]).argmax())
            out.append((cid, width, area, pred_label))
    return out


@torch.no_grad()
def collect(model, loader, class_ids, min_area, device):
    records = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        gts = batch["mask"].numpy().astype(np.int32)
        preds = model(images).argmax(1).cpu().numpy().astype(np.int32)
        for gt, pred in zip(gts, preds):
            records.extend(instance_records(gt, pred, class_ids, min_area))
    return records


def report(records, class_ids, names, n_bins):
    widths = np.array([r[1] for r in records])
    if len(widths) == 0:
        print("No instances found.")
        return None
    edges = np.unique(np.percentile(widths, np.linspace(0, 100, n_bins + 1)))
    edges[-1] += 1e-6

    print("\n" + "=" * 78)
    print(" INSTANCE ACCURACY BY APPARENT WIDTH")
    print("=" * 78)
    curves = {}
    for cid in class_ids:
        rows = [r for r in records if r[0] == cid]
        if not rows:
            continue
        w = np.array([r[1] for r in rows])
        ok = np.array([r[3] == cid for r in rows])
        print(f"\n{names[cid]}  (n={len(rows)}, overall instance accuracy {ok.mean():.3f})")
        print(f"  {'width range':<20}{'n':>7}{'accuracy':>11}   most common error")
        xs, ys = [], []
        for i in range(len(edges) - 1):
            sel = (w >= edges[i]) & (w < edges[i + 1])
            if sel.sum() == 0:
                continue
            acc = ok[sel].mean()
            wrong = [r[3] for r, s in zip(rows, sel) if s and r[3] != cid]
            if wrong:
                vals, cnt = np.unique(wrong, return_counts=True)
                worst = f"{names[vals[cnt.argmax()]]} ({cnt.max()}/{int(sel.sum())})"
            else:
                worst = "-"
            print(f"  {edges[i]:7.1f} - {edges[i+1]:7.1f}{int(sel.sum()):>7}{acc:>11.3f}   {worst}")
            xs.append(0.5 * (edges[i] + edges[i + 1]))
            ys.append(acc)
        curves[cid] = (xs, ys)
    return curves


def verdict(records, class_ids, names, band):
    """Compare accuracy inside the ambiguous width band with accuracy outside it."""
    lo, hi = band
    print("\n" + "-" * 78)
    print(f"Ambiguous band = {lo:.1f} - {hi:.1f} px (where the two classes overlap)")
    inside_ok, inside_n, outside_ok, outside_n = 0, 0, 0, 0
    for cid in class_ids:
        rows = [r for r in records if r[0] == cid]
        for _, w, _, p in rows:
            hit = int(p == cid)
            if lo <= w <= hi:
                inside_ok += hit
                inside_n += 1
            else:
                outside_ok += hit
                outside_n += 1
    acc_in = inside_ok / max(inside_n, 1)
    acc_out = outside_ok / max(outside_n, 1)
    print(f"  accuracy INSIDE  the band : {acc_in:.3f}  (n={inside_n})")
    print(f"  accuracy OUTSIDE the band : {acc_out:.3f}  (n={outside_n})")
    print("-" * 78)
    gap = acc_out - acc_in
    if acc_out >= 0.85 and gap >= 0.15:
        print("READ -> A: the model is accurate away from the boundary and fails only")
        print("          inside it. The class boundary itself is the bottleneck.")
    elif acc_out < 0.75:
        print("READ -> B: the model is also wrong on instances whose width is")
        print("          unambiguous. It is not exploiting the width cue.")
    else:
        print("READ -> mixed: part boundary ambiguity, part model. See the table above:")
        print("          check whether wide instances are still misread.")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint_path, device)
    class_names = ckpt.get("class_names") or get_class_names(args.variant)
    model_name = ckpt.get("model_name", "unet_mobilenet")
    img_size = args.img_size or ckpt.get("img_size", 512)
    class_ids = [int(c) for c in args.classes.split(",")]

    model = build_model(model_name, 3, len(class_names), pretrained=False,
                        head=ckpt.get("head", "flat"), class_names=class_names).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: {model_name} | img_size {img_size} | device {device}")
    print("Classes: " + ", ".join(f"{c}={class_names[c]}" for c in class_ids))

    records = []
    for split in args.splits:
        if resolve_split_dirs(args.data_dir, split) is None:
            print(f"  (skipping '{split}')")
            continue
        loader = build_loader(args.data_dir, split, img_size, args.batch_size,
                              args.num_workers, False)
        print(f"  {split}: {len(loader.dataset)} images")
        records.extend(collect(model, loader, class_ids, args.min_area, device))

    if not records:
        print("No instances found.")
        return
    print(f"\nTotal ground-truth instances: {len(records)}")

    curves = report(records, class_ids, class_names, args.bins)
    band = tuple(float(x) for x in args.band.split(","))
    verdict(records, class_ids, class_names, band)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "boundary_errors.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("gt_class,gt_name,width_px,area_px,pred_class,pred_name,correct\n")
        for cid, w, a, p in records:
            f.write(f"{cid},{class_names[cid]},{w:.3f},{a},{p},{class_names[p]},{int(p == cid)}\n")
    print(f"\nPer-instance table: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        for cid, (xs, ys) in (curves or {}).items():
            plt.plot(xs, ys, marker="o", label=class_names[cid])
        plt.axvspan(band[0], band[1], color="grey", alpha=0.2, label="ambiguous band")
        plt.ylim(0, 1)
        plt.xlabel("apparent width of the ground-truth instance (px)")
        plt.ylabel("instance accuracy")
        plt.title("Is the error concentrated at the class boundary?")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig = os.path.join(args.out_dir, "accuracy_vs_width.png")
        plt.savefig(fig, dpi=150)
        print(f"Figure: {fig}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--variant", default="7class")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    p.add_argument("--classes", default="3,7")
    p.add_argument("--min_area", type=int, default=50)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--band", default="8,20",
                   help="width range where the two classes overlap, 'lo,hi'")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", default="runs/analysis")
    main(p.parse_args())
