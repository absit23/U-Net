"""Boundary-tolerant IoU: is a class weak because the model misses it, or because
its thin/ambiguous boundaries are a few pixels off?

    python tolerance_eval.py --data_dir data\\Try3X_811_ds --checkpoint_path runs\\control\\img640_e120\\last_unet_mobilenet.pth --split test

For each class and each tolerance k (pixels, at the model's input resolution):
  matched_pred = predicted pixels lying within k px of a ground-truth pixel of that class
  matched_gt   = ground-truth pixels lying within k px of a predicted pixel of that class
  TP' = (matched_pred + matched_gt) / 2 ,  FP' = |pred| - matched_pred ,  FN' = |gt| - matched_gt
  tolerant IoU = TP' / (TP' + FP' + FN')
k = 0 is the ordinary IoU. Report BOTH the ordinary and the tolerant numbers: the tolerant one is
optimistic by design, it only says how much of the error is boundary placement.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs import get_class_names
from datasets import build_loader, resolve_split_dirs
from models import build_model
from training.checkpoint import load_checkpoint


def dilate(x, k):
    """x: (B,C,H,W) float 0/1. Square dilation by k pixels (Chebyshev distance)."""
    if k == 0:
        return x
    return F.max_pool2d(x, kernel_size=2 * k + 1, stride=1, padding=k)


def update_stats(stats, pred, gt, num_classes, ks):
    """Accumulate pixel counts. pred, gt: (B,H,W) long."""
    P = F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
    G = F.one_hot(gt, num_classes).permute(0, 3, 1, 2).float()
    for k in ks:
        Pd, Gd = dilate(P, k), dilate(G, k)
        stats[k]["pred"] += P.sum((0, 2, 3)).double()
        stats[k]["gt"] += G.sum((0, 2, 3)).double()
        stats[k]["m_pred"] += (P * Gd).sum((0, 2, 3)).double()   # pred pixels near a GT pixel
        stats[k]["m_gt"] += (G * Pd).sum((0, 2, 3)).double()     # GT pixels near a pred pixel


def tolerant_iou(s):
    tp = (s["m_pred"] + s["m_gt"]) / 2
    fp = s["pred"] - s["m_pred"]
    fn = s["gt"] - s["m_gt"]
    return tp / (tp + fp + fn).clamp(min=1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint_path, device)
    class_names = ckpt.get("class_names") or get_class_names(args.variant)
    model_name = ckpt.get("model_name", args.model_name)
    img_size = args.img_size or ckpt.get("img_size", 512)
    nc = len(class_names)

    if not resolve_split_dirs(args.data_dir, args.split):
        raise FileNotFoundError(f"Split '{args.split}' not found under '{args.data_dir}'")
    loader = build_loader(args.data_dir, args.split, img_size, args.batch_size, args.num_workers, False)

    model = build_model(model_name, 3, nc, pretrained=False,
                        head=ckpt.get("head", "flat"), class_names=class_names).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ks = sorted(set([0] + args.k))
    z = lambda: torch.zeros(nc, dtype=torch.float64, device=device)
    stats = {k: {"pred": z(), "gt": z(), "m_pred": z(), "m_gt": z()} for k in ks}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Tolerance eval ({args.split})"):
            x, y = batch["image"].to(device), batch["mask"].to(device)
            pred = model(x).argmax(1)
            update_stats(stats, pred, y, nc, ks)

    res = {k: tolerant_iou(stats[k]).cpu() for k in ks}
    print(f"\nSplit: {args.split} | input size: {img_size} px | k = tolerance in pixels (k=0 is the ordinary IoU)\n")
    print(f"{'Class':<22}" + "".join(f"k={k:<7}" for k in ks))
    for c in range(1, nc):
        print(f"{class_names[c]:<22}" + "".join(f"{res[k][c].item():<9.4f}" for k in ks))
    print(f"{'mIoU (damage classes)':<22}" + "".join(f"{res[k][1:].mean().item():<9.4f}" for k in ks))
    print("\nReport both rows: k=0 is the standard number; k>0 only shows how much error is boundary placement.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Boundary-tolerant IoU per class")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--variant", default="7class", choices=["6class", "7class", "merged6", "orig8"])
    p.add_argument("--model_name", default="unet_mobilenet")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    main(p.parse_args())
