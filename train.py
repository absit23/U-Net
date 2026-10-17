"""Train a concrete-damage segmentation model.

Colab example:
    from google.colab import drive; drive.mount('/content/drive')
    !python train.py --data_dir /content/dataset --variant 7class \
        --runs_dir "/content/drive/MyDrive/U-Net/runs"
"""
import argparse
import os

import torch

from configs import get_class_names
from datasets import build_loader
from models import build_model, print_model_summary
from models.builder import AVAILABLE_MODELS
from training.checkpoint import best_path, last_path, save_checkpoint, load_checkpoint
from training.class_weights import resolve_class_weights
from training.loops import train_one_epoch, evaluate_loader
from training.losses import WeightedCombinedLoss
from training.metrics import SegmentationMetrics
from training.tracker import ExperimentTracker, get_next_exp_name, new_history, print_results


def parse_args():
    p = argparse.ArgumentParser(description="Train concrete-damage segmentation models")
    p.add_argument("--data_dir", required=True, help="Dataset root with train/ and valid/")
    p.add_argument("--variant", default="7class", choices=["6class", "7class"])
    p.add_argument("--model_name", default="unet_mobilenet", choices=AVAILABLE_MODELS)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--weight_ce", type=float, default=1.0)
    p.add_argument("--weight_dice", type=float, default=1.0)
    p.add_argument("--class_weights", default=None,
                   help="None = manual defaults | 'auto' = from train masks | 'w0,w1,...' in class-ID order")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p.add_argument("--runs_dir", default="runs", help="Where experiment folders are saved (use a Drive path on Colab)")
    p.add_argument("--exp_name", default=None, help="Experiment folder; default = next roundN")
    return p.parse_args()


def main(args):
    class_names = get_class_names(args.variant)
    num_classes = len(class_names)

    if not args.exp_name:
        args.exp_name = (os.path.basename(os.path.dirname(os.path.abspath(args.resume)))
                         if args.resume else get_next_exp_name(args.runs_dir))
    exp_dir = os.path.join(args.runs_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Variant: {args.variant} ({num_classes} IDs) | Output: {exp_dir}\n")

    tracker = ExperimentTracker(exp_dir, args, class_names)
    pin = device.type == "cuda"
    train_loader = build_loader(args.data_dir, "train", args.img_size, args.batch_size, args.num_workers, True, pin)
    val_loader = build_loader(args.data_dir, "valid", args.img_size, args.batch_size, args.num_workers, False, pin)

    model = build_model(args.model_name, 3, num_classes, args.pretrained).to(device)
    print_model_summary(model, args.model_name, args.batch_size, args.img_size, device)

    weights = resolve_class_weights(args.class_weights, class_names, args.data_dir)
    criterion = WeightedCombinedLoss(num_classes, args.weight_ce, args.weight_dice,
                                     torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metrics = SegmentationMetrics(num_classes)

    history, best_miou, best_epoch, best_metrics, start_epoch = new_history(), 0.0, 0, {}, 1

    resume_path = args.resume or last_path(exp_dir, args.model_name)
    if os.path.isfile(resume_path):
        ckpt = load_checkpoint(resume_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_miou = float(ckpt.get("best_miou", 0.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        best_metrics = ckpt.get("best_metrics", {})
        history = ckpt.get("history", history)
        print(f"=> Resumed from '{resume_path}' at epoch {start_epoch} (best mIoU {best_miou:.4f})\n")

    if start_epoch > args.epochs:
        print(f"Checkpoint already at epoch {start_epoch - 1} >= --epochs {args.epochs}. Increase --epochs to continue.")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate_loader(model, val_loader, criterion, metrics, device)
        scheduler.step()
        tracker.log_epoch(epoch, train_loss, val, history)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train {train_loss:.4f} | Val {val['loss']:.4f} | "
              f"mIoU {val['mIoU']:.4f} | F1 {val['mDice']:.4f} | P {val['mPrecision']:.4f} | R {val['mRecall']:.4f}")

        is_best = val["mIoU"] > best_miou
        if is_best:
            best_miou, best_epoch, best_metrics = float(val["mIoU"]), epoch, val

        state = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou, "best_epoch": best_epoch, "best_metrics": best_metrics,
            "history": history, "variant": args.variant, "class_names": class_names,
            "model_name": args.model_name, "img_size": args.img_size, "class_weights": weights,
        }
        save_checkpoint(last_path(exp_dir, args.model_name), state)
        if is_best:
            save_checkpoint(best_path(exp_dir, args.model_name), state)
            tracker.save_confusion_matrix(val["confusion_matrix"])
            print(f"  -> new best saved (mIoU {best_miou:.4f})")

    if best_metrics:
        print_results(f"BEST EPOCH {best_epoch}/{args.epochs}  ({args.exp_name})", best_metrics, class_names)


if __name__ == "__main__":
    main(parse_args())
