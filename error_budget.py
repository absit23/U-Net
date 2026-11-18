"""Where do the errors actually come from: the background, or other classes?

Reads a confusion matrix saved by evaluate.py (rows = ground truth, columns =
prediction) and splits every class's error budget into

    background errors : damage missed as background, or background called damage
    class conflict    : damage confused with ANOTHER damage class

and then ranks the class pairs by how much each one costs, measured as the mIoU
the model would gain if that pair were merged into a single class.

    python error_budget.py --cm runs/BASELINE_v1/confusion_matrix_test.csv
"""
import argparse
import os

import numpy as np


def load_cm(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    names = [n.strip() for n in header.split(",")]
    cm = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if cm.shape[0] != len(names):
        raise ValueError(f"{cm.shape[0]} rows but {len(names)} class names in the header.")
    return cm, names


def iou_vector(cm):
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    den = tp + fp + fn
    return np.divide(tp, den, out=np.full(len(tp), np.nan), where=den > 0)


def merged_miou(cm, i, j):
    """mIoU over damage classes after merging classes i and j into one."""
    keep = [k for k in range(cm.shape[0]) if k != j]
    m = cm.copy()
    m[i, :] += m[j, :]
    m[:, i] += m[:, j]
    m = m[np.ix_(keep, keep)]
    return float(np.nanmean(iou_vector(m)[1:]))


def main(args):
    cm, names = load_cm(args.cm)
    n = len(names)
    dmg = list(range(1, n))                     # damage classes, background = 0
    iou = iou_vector(cm)
    base_miou = float(np.nanmean(iou[dmg]))

    print("=" * 86)
    print(f" ERROR BUDGET  ({os.path.basename(args.cm)})   mIoU = {base_miou:.4f}")
    print("=" * 86)
    print("For each class: of everything the model got wrong about it (missed pixels +")
    print("wrongly claimed pixels), how much involved the background, and how much was")
    print("a mix-up with another damage class.\n")
    print(f"{'Class':<20}{'IoU':>7}{'errors':>11}{'background':>12}{'conflict':>10}   worst partner")
    print("-" * 86)

    tot_bg, tot_cls = 0.0, 0.0
    for c in dmg:
        tp = cm[c, c]
        fn_bg = cm[c, 0]
        fn_cls = cm[c, dmg].sum() - tp
        fp_bg = cm[0, c]
        fp_cls = cm[dmg, c].sum() - tp
        err = fn_bg + fn_cls + fp_bg + fp_cls
        if err == 0:
            continue
        bg_share = (fn_bg + fp_bg) / err
        cls_share = (fn_cls + fp_cls) / err
        tot_bg += fn_bg + fp_bg
        tot_cls += fn_cls + fp_cls

        pair = [(cm[c, k] + cm[k, c], k) for k in dmg if k != c]
        worst_v, worst_k = max(pair) if pair else (0, None)
        worst = f"{names[worst_k]} ({worst_v / err:.0%} of errors)" if worst_k is not None else "-"
        print(f"{names[c]:<20}{iou[c]:>7.3f}{err:>11.0f}{bg_share:>11.0%}{cls_share:>10.0%}   {worst}")

    tot = tot_bg + tot_cls
    print("-" * 86)
    print(f"{'ALL DAMAGE CLASSES':<20}{'':>7}{tot:>11.0f}{tot_bg / tot:>11.0%}{tot_cls / tot:>10.0%}")

    print("\n" + "=" * 86)
    print(" CLASS PAIRS, RANKED BY WHAT THEY COST")
    print("=" * 86)
    print("'mIoU if merged' = the score the model would reach if the pair were one class.")
    print("A large gain means the two classes are fighting each other.\n")
    print(f"{'pair':<44}{'confused px':>13}{'mIoU if merged':>16}{'gain':>8}")
    print("-" * 86)
    rows = []
    for a_i, a in enumerate(dmg):
        for b in dmg[a_i + 1:]:
            conf = cm[a, b] + cm[b, a]
            rows.append((merged_miou(cm, a, b) - base_miou, conf, a, b))
    for gain, conf, a, b in sorted(rows, reverse=True)[:args.top]:
        print(f"{names[a] + '  <->  ' + names[b]:<44}{conf:>13.0f}"
              f"{base_miou + gain:>16.4f}{gain:>+8.4f}")
    median_gain = float(np.median([g for g, _, _, _ in rows]))
    print("-" * 86)
    print(f"Reference: merging ANY two classes gains about {median_gain:+.4f} on average, simply")
    print("because one class leaves the mean. Only pairs clearly above that are real conflicts.")

    print("\n" + "=" * 86)
    print(" READING IT")
    print("=" * 86)
    bgp = tot_bg / tot
    if bgp > 0.65:
        print(" Most of the error is against the BACKGROUND: the model misses damage or")
        print(" invents it. That is a detection / extent problem, not a taxonomy problem.")
    elif bgp < 0.35:
        print(" Most of the error is BETWEEN damage classes: the model finds the damage")
        print(" but labels it as the wrong one. That is a taxonomy problem.")
    else:
        print(f" Mixed: {bgp:.0%} background, {1 - bgp:.0%} class conflict. Look at the")
        print(" per-class rows above; the two problems hit different classes.")
    print(" The pair table says whether one pair dominates or the conflict is spread out.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cm", required=True, help="confusion_matrix_test.csv saved by evaluate.py")
    p.add_argument("--top", type=int, default=10, help="how many pairs to list")
    main(p.parse_args())
