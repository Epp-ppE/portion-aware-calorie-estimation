from collections import defaultdict

import torch

from src.data import TARGET_NAMES


def predict_dish_level(model, loader, normalizer, device):
    """Run the model over every frame in loader, denormalize predictions
    back to real units, then average all of a dish's frame predictions into
    one prediction per dish (since that's how dishes are actually scored).

    Returns dish_id -> tensor of shape (4,) in real units.
    """
    model.eval()
    sums = defaultdict(lambda: torch.zeros(len(TARGET_NAMES)))
    counts = defaultdict(int)

    with torch.no_grad():
        for images, _, dish_ids in loader:
            images = images.to(device)
            preds_norm = model(images)
            preds = normalizer.denormalize(preds_norm).cpu()
            for pred, dish_id in zip(preds, dish_ids):
                sums[dish_id] += pred
                counts[dish_id] += 1

    return {dish_id: sums[dish_id] / counts[dish_id] for dish_id in sums}


def true_dish_targets(dish_ids, metadata):
    """dish_id -> true label tensor of shape (4,), same order as predict_dish_level."""
    return {
        dish_id: torch.tensor([metadata[dish_id][name] for name in TARGET_NAMES], dtype=torch.float32)
        for dish_id in dish_ids
        if dish_id in metadata
    }


def compute_mae(dish_preds, dish_targets):
    """dish_preds, dish_targets: dicts with the same dish_id keys.

    Returns {target_name: (mae, mae_percent_of_mean_true)}.
    """
    dish_ids = [d for d in dish_targets if d in dish_preds]
    preds = torch.stack([dish_preds[d] for d in dish_ids])
    trues = torch.stack([dish_targets[d] for d in dish_ids])

    abs_error = (preds - trues).abs().mean(dim=0)
    mean_true = trues.mean(dim=0)

    results = {}
    for i, name in enumerate(TARGET_NAMES):
        mae = abs_error[i].item()
        mae_percent = mae / mean_true[i].item() * 100
        results[name] = (mae, mae_percent)
    return results
