import os

import torch


def best_path(exp_dir, model_name):
    return os.path.join(exp_dir, f"best_{model_name}.pth")


def last_path(exp_dir, model_name):
    return os.path.join(exp_dir, f"last_{model_name}.pth")


def save_checkpoint(path, state):
    torch.save(state, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)
