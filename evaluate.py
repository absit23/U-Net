"""Evaluate a saved checkpoint on the test split (falls back to valid).

    python evaluate.py --data_dir /content/dataset --checkpoint_path runs/round1/best_unet_mobilenet.pth
Class names / variant / model are read from the checkpoint.
"""
import argparse
import os

import torch

from configs import get_class_names
from datasets import build_loader, resolve_split_dirs
from models import build_model
from training.checkpoint import load_checkpoint
from training.class_weights import resolve_class_weights
from training.loops import evaluate_loader
from training.losses import WeightedCombinedLoss
from training.metrics import SegmentationMetrics
from training.tracker import print_results, save_confusion_matrix


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: '{args.checkpoint_path}'")
    ckpt = load_checkpoint(args.checkpoint_path, device)

    class_names = ckpt.get("class_names") or get_class_names(args.variant)
    model_name = ckpt.get("model_name", args.model_name)
    img_size = args.img_size or ckpt.get("img_size", 512)
    num_classes = len(class_names)

    split = "test" if resolve_split_dirs(args.data_dir, "test") else "valid"
    if split == "valid":
        print("WARNING: no test split found; evaluating on 'valid'. Do not report this as a test result.")
    loader = build_loader(args.data_dir, split, img_size, args.batch_size, args.num_workers, False)

    model = build_model(model_name, 3, num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Same (weighted) loss as training so test loss is comparable with val loss.
    saved = ckpt.get("class_weights") or resolve_class_weights(None, class_names, args.data_dir)
    weights = torch.tensor(saved, dtype=torch.float32, device=device)
    criterion = WeightedCombinedLoss(num_classes, class_weights=weights)

    m = evaluate_loader(model, loader, criterion, SegmentationMetrics(num_classes), device, desc=f"Eval ({split})")
    print_results(f"FINAL EVALUATION [{split}]: {model_name.upper()}  (loss {m['loss']:.4f})", m, class_names)

    out_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
    save_confusion_matrix(m["confusion_matrix"], class_names, out_dir, prefix=f"confusion_matrix_{split}")
    print(f"Confusion matrix saved to: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a concrete-damage segmentation checkpoint")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--variant", default="7class", choices=["6class", "7class"], help="Only used if the checkpoint has no class names")
    p.add_argument("--model_name", default="unet_mobilenet", help="Only used if the checkpoint has no model name")
    p.add_argument("--img_size", type=int, default=None, help="Default: value stored in the checkpoint")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
