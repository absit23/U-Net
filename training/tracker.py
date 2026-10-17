"""Experiment logging: args.json, metrics.csv, training curves, confusion matrix."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless (Colab-safe)
import matplotlib.pyplot as plt
import numpy as np


def get_next_exp_name(runs_dir, prefix="round"):
    os.makedirs(runs_dir, exist_ok=True)
    nums = [int(d[len(prefix):]) for d in os.listdir(runs_dir)
            if d.startswith(prefix) and d[len(prefix):].isdigit()
            and os.path.isdir(os.path.join(runs_dir, d))]
    return f"{prefix}{max(nums, default=0) + 1}"


def new_history():
    return {k: [] for k in ("epoch", "train_loss", "val_loss", "val_mIoU", "val_mDice",
                            "val_mPrecision", "val_mRecall", "per_class_iou")}


class ExperimentTracker:
    def __init__(self, exp_dir, args, class_names):
        self.exp_dir = exp_dir
        self.class_names = class_names
        self.csv_path = os.path.join(exp_dir, "metrics.csv")
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    def log_epoch(self, epoch, train_loss, m, history):
        history["epoch"].append(int(epoch))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(m["loss"]))
        history["val_mIoU"].append(float(m["mIoU"]))
        history["val_mDice"].append(float(m["mDice"]))
        history["val_mPrecision"].append(float(m["mPrecision"]))
        history["val_mRecall"].append(float(m["mRecall"]))
        history["per_class_iou"].append([float(x) for x in m["per_class_iou"]])

        new_file = not os.path.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_mDice",
                            "val_mPrecision", "val_mRecall"] + [f"IoU_{c}" for c in self.class_names])
            w.writerow([epoch, f"{train_loss:.4f}", f"{m['loss']:.4f}", f"{m['mIoU']:.4f}",
                        f"{m['mDice']:.4f}", f"{m['mPrecision']:.4f}", f"{m['mRecall']:.4f}"]
                       + [f"{x:.4f}" for x in m["per_class_iou"]])
        self.plot_curves(history)

    def _save_fig(self, name):
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, name), dpi=300)
        plt.close()

    def plot_curves(self, h):
        ep = h["epoch"]
        plt.figure(figsize=(8, 5))
        plt.plot(ep, h["train_loss"], label="Train loss", linewidth=2)
        plt.plot(ep, h["val_loss"], label="Val loss", linewidth=2)
        plt.title("Training & validation loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
        self._save_fig("loss_curve.png")

        plt.figure(figsize=(8, 5))
        for key, label in (("val_mIoU", "mIoU"), ("val_mDice", "mDice (F1)"),
                           ("val_mPrecision", "Precision"), ("val_mRecall", "Recall")):
            plt.plot(ep, h[key], label=label, linewidth=2)
        plt.title("Validation metrics (excl. background)"); plt.xlabel("Epoch"); plt.ylabel("Score")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
        self._save_fig("metrics_curve.png")

        plt.figure(figsize=(10, 6))
        for i, name in enumerate(self.class_names):
            if i == 0:
                continue
            plt.plot(ep, [row[i] for row in h["per_class_iou"]], label=name, linewidth=1.8)
        plt.title("Per-class IoU over epochs"); plt.xlabel("Epoch"); plt.ylabel("IoU")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left"); plt.grid(True, linestyle="--", alpha=0.6)
        self._save_fig("per_class_iou_curve.png")

    def save_confusion_matrix(self, cm, prefix="confusion_matrix"):
        """Save raw CSV + row-normalised PNG (rows = ground truth)."""
        save_confusion_matrix(cm, self.class_names, self.exp_dir, prefix)


def save_confusion_matrix(cm, class_names, out_dir, prefix="confusion_matrix"):
    cm = np.asarray(cm, dtype=float)
    np.savetxt(os.path.join(out_dir, f"{prefix}.csv"), cm.astype(np.int64), fmt="%d",
               delimiter=",", header=",".join(class_names), comments="")
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    plt.figure(figsize=(1.0 * n + 3, 1.0 * n + 2))
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(n), class_names, rotation=45, ha="right"); plt.yticks(range(n), class_names)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                     color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    plt.xlabel("Predicted"); plt.ylabel("Ground truth"); plt.title("Confusion matrix (row-normalised)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}.png"), dpi=300)
    plt.close()


def print_results(title, m, class_names):
    print("\n" + "=" * 70 + f"\n {title}\n" + "=" * 70)
    print(f"mIoU {m['mIoU']:.4f} | F1 {m['mDice']:.4f} | Precision {m['mPrecision']:.4f} | Recall {m['mRecall']:.4f}")
    print(f"\n{'ID':<4} | {'Class':<20} | {'IoU':<8} | {'Dice':<8}")
    print("-" * 48)
    for i, name in enumerate(class_names):
        print(f"{i:<4} | {name:<20} | {m['per_class_iou'][i]:<8.4f} | {m['per_class_dice'][i]:<8.4f}")
    print("=" * 70 + "\n")
