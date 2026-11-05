"""Image-level severity from segmentation masks (no training needed).

For each image, a severity level is computed from the ground-truth mask and from the
model's predicted mask, then the two are compared. It works with flat 7-class
checkpoints (Delamination + Rebar corrosion are treated as Severe) and with the
merged6 checkpoint, because classes are matched by name.

Three rules are reported:
  A) Yasmin-style (thesis, Eq. 3.1): ratio = severe pixels / image pixels, then
     Less (<40%), Medium (40-69%), Very (>=70%).
  B) Worst level present: Severe if severe area >= t of the image, else Moderate if
     moderate area >= t, else Minor if minor area >= t, else None (t = --min_area).
  C) Dominant level: if total spalling area >= t, the level (Minor / Moderate / Severe)
     with the largest area; else None. Less sensitive to a small stray patch than B.
Also reported: error of the severe-area ratio itself (percentage points), and Rules B/C
restricted to images that really contain spalling (ground truth), because most of the
'None' images are crack / bleeding images that are trivially easy.

Example (each --run takes: label, checkpoint, data_dir):
    python image_severity.py --splits test ^
      --run flat1 runs\\control\\w_sqrt\\last_unet_mobilenet.pth data\\dataset_v1 ^
      --run flat2 runs\\control\\w_sqrt_b\\last_unet_mobilenet.pth data\\dataset_v1 ^
      --run m6 runs\\control\\m6_sqrt\\last_unet_mobilenet.pth data\\dataset_v1_merged
"""
import argparse
import csv

import numpy as np

SEVERE = {"Severe spalling", "Delamination", "Rebar corrosion"}
MODERATE = {"Moderate spalling"}
MINOR = {"Minor spalling"}
LABELS_A = ["Less(<40%)", "Medium", "Very(>=70%)"]
LABELS_B = ["None", "Minor", "Moderate", "Severe"]


def family_ratios(cms, names):
    """cms: (N, n, n) per-image confusion [gt, pred]. Returns gt/pred ratio dicts."""
    total = cms.sum((1, 2)).astype(float)
    out = {}
    for tag, counts in (("gt", cms.sum(2)), ("pr", cms.sum(1))):
        out[tag] = {}
        for fam, members in (("sev", SEVERE), ("mod", MODERATE), ("min", MINOR)):
            idx = [i for i, n in enumerate(names) if n in members]
            out[tag][fam] = counts[:, idx].sum(1) / total
    return out


def level_a(r):
    s = r["sev"]
    return np.where(s < 0.40, 0, np.where(s < 0.70, 1, 2))


def level_b(r, t):
    return np.where(r["sev"] >= t, 3, np.where(r["mod"] >= t, 2, np.where(r["min"] >= t, 1, 0)))


def level_c(r, t):
    total = r["sev"] + r["mod"] + r["min"]
    dom = np.stack([r["min"], r["mod"], r["sev"]], 1).argmax(1) + 1
    return np.where(total >= t, dom, 0)


def mae_report(title, gt_r, ratios, runs, mask, rng, n_boot):
    idx_all = np.flatnonzero(mask)
    idx_sets = [rng.choice(idx_all, len(idx_all)) for _ in range(n_boot)]
    print(f"\n===== {title} ({len(idx_all)} images) =====")
    for l, _, _ in runs:
        line = f"{l:<8}"
        for fam, nm in (("sev", "severe"), ("mod", "moderate")):
            err = np.abs(ratios[l]["pr"][fam] - gt_r[fam]) * 100
            lo, hi = ci([err[i].mean() for i in idx_sets])
            line += f"  {nm} ratio MAE {err[idx_all].mean():5.2f} pts [{lo:.2f}, {hi:.2f}]"
        print(line)


def qwk(gt, pr, k):
    o = np.zeros((k, k))
    np.add.at(o, (gt, pr), 1)
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    den = (w * e).sum()
    return np.nan if den == 0 else 1 - (w * o).sum() / den


def stats(gt, pr, k):
    return {"accuracy": np.mean(gt == pr), "within 1 level": np.mean(np.abs(gt - pr) <= 1),
            "over-graded": np.mean(pr > gt), "under-graded": np.mean(pr < gt), "QWK": qwk(gt, pr, k)}


def ci(vals):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)


def report(title, gt, preds, labels, rng, n_boot, full=True):
    k, N = len(labels), len(gt)
    print(f"\n===== {title} =====")
    print("GT distribution: " + ", ".join(f"{l}={int((gt == i).sum())}" for i, l in enumerate(labels)))
    idx_sets = [rng.integers(0, N, N) for _ in range(n_boot)]
    boots = {}
    for name, pr in preds.items():
        s = stats(gt, pr, k)
        b = [stats(gt[i], pr[i], k) for i in idx_sets]
        boots[name] = b
        print(f"\n[{name}]")
        for key, val in s.items():
            lo, hi = ci([x[key] for x in b])
            print(f"  {key:<15} {val:6.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
        if full:
            cm = np.zeros((k, k), int)
            np.add.at(cm, (gt, pr), 1)
            print("  confusion (rows = GT, cols = predicted):")
            print("  " + " " * 12 + "".join(f"{l:>12}" for l in labels))
            for i, l in enumerate(labels):
                print(f"  {l:<12}" + "".join(f"{v:>12d}" for v in cm[i]))
    names = list(preds)
    for other in names[1:]:
        for key in ("accuracy", "QWK"):
            d = [bo[key] - ba[key] for ba, bo in zip(boots[names[0]], boots[other])]
            lo, hi = ci(d)
            base = stats(gt, preds[other], k)[key] - stats(gt, preds[names[0]], k)[key]
            flag = " *" if (lo > 0 or hi < 0) else ""
            print(f"  diff {other} - {names[0]} {key}: {base:+.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")


def run_analysis(runs, files, min_areas, n_boot, seed, out_csv=None):
    """runs: list of (label, cms, names). All runs must cover the same images."""
    rng = np.random.default_rng(seed)
    ratios = {label: family_ratios(cms, names) for label, cms, names in runs}
    first = runs[0][0]
    gt_r = ratios[first]["gt"]
    for label in ratios:  # ground truth must agree between datasets (merged vs flat)
        for fam in gt_r:
            if not np.allclose(ratios[label]["gt"][fam], gt_r[fam]):
                print(f"WARNING: ground-truth {fam} ratios of '{label}' differ from '{first}'.")
    print(f"\nImages: {len(files)}")
    print("GT image share of severe pixels (percentiles 50/75/90/99, %): "
          + ", ".join(f"{100 * np.percentile(gt_r['sev'], p):.2f}" for p in (50, 75, 90, 99)))

    gt_a = level_a(gt_r)
    report("Rule A: Yasmin-style severe-area ratio (thresholds 40% / 70%)", gt_a,
           {l: level_a(ratios[l]["pr"]) for l, _, _ in runs}, LABELS_A, rng, n_boot)
    for j, t in enumerate(min_areas):
        gt_b = level_b(gt_r, t)
        report(f"Rule B: worst level present (min area {100 * t:.1f}% of image)", gt_b,
               {l: level_b(ratios[l]["pr"], t) for l, _, _ in runs}, LABELS_B, rng, n_boot, full=(j == 0))

    t0 = min_areas[0]
    gt_c = level_c(gt_r, t0)
    report(f"Rule C: dominant level (min total spalling {100 * t0:.1f}% of image)", gt_c,
           {l: level_c(ratios[l]["pr"], t0) for l, _, _ in runs}, LABELS_B, rng, n_boot)

    sp = gt_c > 0  # images that really contain spalling (from ground truth only)
    print(f"\nImages with spalling in the ground truth: {int(sp.sum())} of {len(files)}")
    report("Rule B on spalling images only", level_b(gt_r, t0)[sp],
           {l: level_b(ratios[l]["pr"], t0)[sp] for l, _, _ in runs}, LABELS_B, rng, n_boot)
    report("Rule C on spalling images only", gt_c[sp],
           {l: level_c(ratios[l]["pr"], t0)[sp] for l, _, _ in runs}, LABELS_B, rng, n_boot)
    mae_report("Error of the area ratio, all images", gt_r, ratios, runs, np.ones(len(files), bool), rng, n_boot)
    mae_report("Error of the area ratio, spalling images only", gt_r, ratios, runs, sp, rng, n_boot)

    if out_csv:
        t = min_areas[0]
        header = ["file", "gt_sev", "gt_mod", "gt_min", "gt_levelA", f"gt_levelB_t{t}", f"gt_levelC_t{t}"]
        for l, _, _ in runs:
            header += [f"{l}_sev", f"{l}_mod", f"{l}_min", f"{l}_levelA", f"{l}_levelB", f"{l}_levelC"]
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            la = {l: level_a(ratios[l]["pr"]) for l, _, _ in runs}
            lb = {l: level_b(ratios[l]["pr"], t) for l, _, _ in runs}
            lc = {l: level_c(ratios[l]["pr"], t) for l, _, _ in runs}
            for i, name in enumerate(files):
                row = [name, *(f"{gt_r[x][i]:.5f}" for x in ("sev", "mod", "min")), gt_a[i], level_b(gt_r, t)[i], gt_c[i]]
                for l, _, _ in runs:
                    p = ratios[l]["pr"]
                    row += [f"{p['sev'][i]:.5f}", f"{p['mod'][i]:.5f}", f"{p['min'][i]:.5f}", la[l][i], lb[l][i], lc[l][i]]
                w.writerow(row)
        print(f"\nPer-image table saved to: {out_csv}")


def main(args):
    import torch
    from compare_runs import per_image_confusions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs, files0 = [], None
    for label, ckpt, data_dir in args.run:
        cms, names, files = per_image_confusions(ckpt, data_dir, device, args.splits)
        if files0 is None:
            files0 = files
        assert files == files0, f"'{label}' evaluates a different image list."
        runs.append((label, cms, names))
        print(f"{label}: {len(files)} images, classes = {names}")
    run_analysis(runs, files0, args.min_area, args.n_boot, args.seed, args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs=3, action="append", metavar=("LABEL", "CKPT", "DATA_DIR"), required=True,
                   help="Repeat for each model. The first run is the reference for differences.")
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--min_area", nargs="+", type=float, default=[0.01, 0.005, 0.02],
                   help="Rule B minimum area fractions; the first one gets the full report")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="image_severity.csv")
    main(p.parse_args())
