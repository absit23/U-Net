import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    bar = tqdm(loader, desc="Training", leave=False)
    for batch in bar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()

        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, metrics, device, desc="Validation"):
    """Run the model over a loader; return metrics dict incl. 'loss'."""
    model.eval()
    metrics.reset()
    total = 0.0
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images)
        total += criterion(outputs, masks).item()
        metrics.update(outputs, masks)
    result = metrics.compute(include_background=False)
    result["loss"] = total / len(loader)
    return result
