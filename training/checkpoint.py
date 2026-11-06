import os

import torch


def best_path(exp_dir, model_name):
    return os.path.join(exp_dir, f"best_{model_name}.pth")


def last_path(exp_dir, model_name):
    return os.path.join(exp_dir, f"last_{model_name}.pth")


def save_checkpoint(path, state):
    """Atomic save: write to a temp file, flush to disk, then rename over the target.
    If the machine sleeps or loses power mid-write, the previous checkpoint stays intact."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)
