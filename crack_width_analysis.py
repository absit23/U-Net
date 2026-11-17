"""Measure apparent crack width per annotated instance, to test whether the
Microcrack / Structural crack boundary is separable by width at all.

For every connected component of a target class in the ground-truth masks we
compute the Euclidean distance transform (EDT). For a long thin strip of width
w the EDT values are uniform on [0, w/2], so:

    width_med = 4 * median(EDT)      (typical width of the instance)
    width_max = 2 * max(EDT)         (widest point of the instance)

Widths are reported in pixels and as a percentage of the image width, because
pixel width depends on image resolution. Note that apparent width also depends
on camera distance, which we cannot recover -- but apparent width is exactly
what the model sees, so this is the right quantity to test.

Usage:
    python crack_width_analysis.py --data_dir data/Try3X_811_ds
"""
import argparse
import os

import cv2
import numpy as np

from configs import get_class_names


def instance_widths(mask, class_id, min_area):
    """Yield (area, width_med_px, width_max_px) for each component of class_id."""
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    for comp in range(1, n):
        sel = labels == comp
        area = int(sel.sum())
        if area < min_area:
            continue
        # pad so components touching the border are not treated as infinitely wide
        comp_mask = np.zeros_like(binary)
        comp_mask[sel] = 1
        comp_mask = np.pad(comp_mask, 1)
        edt = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        vals = edt[comp_mask > 0]
        yield area, float(4.0 * np.median(vals)), float(2.0 * vals.max())


def collect(data_dir, splits, class_ids, min_area):
    rows = []
    for split in splits:
        mask_dir = os.path.join(data_dir, split, "masks")
        if not os.path.isdir(mask_dir):
            print(f"  (skipping '{split}': no masks folder)")
            continue
        files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(".png"))
        print(f"  {split}: {len(files)} masks")
        for fn in files:
            mask = cv2.imread(os.path.join(mask_dir, fn), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape[:2]
            for cid in class_ids:
                for area, wmed, wmax in instance_widths(mask, cid, min_area):
                    rows.append({
                        "split": split, "file": fn, "class_id": cid,
                        "img_w": w, "img_h": h, "area_px": area,
                        "width_med_px": wmed, "width_max_px": wmax,
                        "width_med_pct": 100.0 * wmed / w,
                        "width_max_pct": 100.0 * wmax / w,
                    })
    return rows


def overlap_coefficient(a, b, bins):
    """0 = distributions fully separated, 1 = identical."""
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(np.minimum(ha, hb).sum())


def best_threshold(a, b):
    """Best single width threshold separating the two classes, and its accuracy.

    'a' is assumed to be the narrower class. Returns (threshold, accuracy,
    baseline) where baseline is the accuracy of always predicting the larger
    class.
    """
    values = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    order = np.argsort(values)
    values, labels = values[order], labels[order]
    best_acc, best_t = 0.0, 0.0
    for i in range(len(values) - 1):
        if values[i] == values[i + 1]:
            continue
        t = 0.5 * (values[i] + values[i + 1])
        acc = ((values <= t) == (labels == 0)).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    baseline = max(len(a), len(b)) / (len(a) + len(b))
    return best_t, float(best_acc), float(baseline)


def main(args):
    names = get_class_names(args.variant)
    class_ids = [int(c) for c in args.classes.split(",")]
    print(f"Classes: " + ", ".join(f"{c}={names[c]}" for c in class_ids))
    print(f"Scanning {args.data_dir} ...")

    rows = collect(args.data_dir, args.splits, class_ids, args.min_area)
    if not rows:
        print("No instances found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "crack_widths.csv")
    keys = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nSaved per-instance table: {csv_path}  ({len(rows)} instances)")

    metric = args.metric
    print("\n" + "=" * 70)
    print(f" APPARENT CRACK WIDTH  ({metric})")
    print("=" * 70)
    print(f"{'class':<22}{'n':>6}{'median':>10}{'mean':>10}{'p10':>10}{'p90':>10}")
    data = {}
    for cid in class_ids:
        vals = np.array([r[metric] for r in rows if r["class_id"] == cid])
        data[cid] = vals
        if len(vals) == 0:
            print(f"{names[cid]:<22}{0:>6}")
            continue
        print(f"{names[cid]:<22}{len(vals):>6}{np.median(vals):>10.2f}"
              f"{vals.mean():>10.2f}{np.percentile(vals, 10):>10.2f}"
              f"{np.percentile(vals, 90):>10.2f}")

    if len(class_ids) == 2 and all(len(data[c]) > 1 for c in class_ids):
        a_id, b_id = class_ids
        a, b = data[a_id], data[b_id]
        if np.median(a) > np.median(b):          # keep 'a' as the narrower class
            a_id, b_id, a, b = b_id, a_id, b, a
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        bins = np.linspace(lo, hi, 40)
        ov = overlap_coefficient(a, b, bins)
        t, acc, base = best_threshold(a, b)

        print("\n" + "-" * 70)
        print(f"Overlap coefficient : {ov:.3f}   (0 = separable, 1 = identical)")
        print(f"Best width threshold: {t:.2f}  -> {names[a_id]} below, {names[b_id]} above")
        print(f"Accuracy of that threshold: {acc:.3f}   (always-majority = {base:.3f})")
        print("-" * 70)
        if ov > 0.5:
            print("READ: the two classes are NOT separable by apparent width.")
        elif ov > 0.25:
            print("READ: widths differ on average but overlap substantially.")
        else:
            print("READ: widths are largely separable; width is not the problem.")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4.5))
            plt.hist(a, bins=bins, alpha=0.6, label=f"{names[a_id]} (n={len(a)})")
            plt.hist(b, bins=bins, alpha=0.6, label=f"{names[b_id]} (n={len(b)})")
            plt.axvline(t, color="k", ls="--", lw=1,
                        label=f"best threshold ({acc:.0%} acc.)")
            plt.xlabel(metric.replace("_", " "))
            plt.ylabel("instances")
            plt.title(f"Apparent width of annotated instances (overlap = {ov:.2f})")
            plt.legend()
            plt.tight_layout()
            fig_path = os.path.join(args.out_dir, "crack_width_hist.png")
            plt.savefig(fig_path, dpi=150)
            print(f"\nFigure saved: {fig_path}")
        except Exception as e:
            print(f"(figure skipped: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--variant", default="7class")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--classes", default="3,7",
                   help="comma-separated class IDs (default 3=Microcrack, 7=Structural crack)")
    p.add_argument("--min_area", type=int, default=50,
                   help="ignore components smaller than this many pixels")
    p.add_argument("--metric", default="width_med_px",
                   choices=["width_med_px", "width_max_px", "width_med_pct", "width_max_pct"])
    p.add_argument("--out_dir", default="runs/analysis")
    main(p.parse_args())
