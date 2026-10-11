from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from src.baselines import aggregate_dish_level, build_embedder, embed_frames, mean_baseline_predict, nearest_neighbor_predict
from src.data import TARGET_NAMES, FrameDataset, build_samples, build_transforms, load_dish_ids, parse_dish_metadata
from src.model import CalorieRegressor, TargetNormalizer
from src.train import FRAMES_ROOT, METADATA_CSV, TEST_SPLIT, TRAIN_SPLIT, CHECKPOINT_OUT


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


def print_mae_table(label, mae_results):
    print(f"\n--- {label} ---")
    for name, (mae, mae_percent) in mae_results.items():
        print(f"{name}: MAE={mae:.2f}  ({mae_percent:.1f}% of mean true value)")


def main():
    """Compare the mean baseline, nearest-neighbor baseline, and the
    trained checkpoint on the test split, side by side."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    metadata = parse_dish_metadata(METADATA_CSV)
    train_ids = load_dish_ids(TRAIN_SPLIT)
    test_ids = load_dish_ids(TEST_SPLIT)
    train_samples = build_samples(FRAMES_ROOT, train_ids, metadata)
    test_samples = build_samples(FRAMES_ROOT, test_ids, metadata)

    # deterministic transforms for everyone here — train frames are used as
    # a lookup index for nearest-neighbor, not for gradient training
    train_eval_set = FrameDataset(train_samples, build_transforms(train=False))
    test_set = FrameDataset(test_samples, build_transforms(train=False))
    train_eval_loader = DataLoader(train_eval_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    dish_trues = true_dish_targets(test_ids, metadata)

    checkpoint = torch.load(CHECKPOINT_OUT, map_location=device, weights_only=True)
    normalizer = TargetNormalizer(checkpoint["means"], checkpoint["stds"])

    # --- mean baseline ---
    mean_preds = mean_baseline_predict(normalizer.means.cpu(), test_ids)
    print_mae_table("mean baseline", compute_mae(mean_preds, dish_trues))

    # --- nearest-neighbor baseline ---
    embedder = build_embedder(device)
    train_embeddings, train_targets, _ = embed_frames(embedder, train_eval_loader, device)
    test_embeddings, _, test_dish_ids = embed_frames(embedder, test_loader, device)
    nn_frame_preds = nearest_neighbor_predict(train_embeddings, train_targets, test_embeddings)
    nn_dish_preds = aggregate_dish_level(nn_frame_preds, test_dish_ids)
    print_mae_table("nearest-neighbor baseline", compute_mae(nn_dish_preds, dish_trues))

    # --- trained model ---
    model = CalorieRegressor().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model_preds = predict_dish_level(model, test_loader, normalizer, device)
    print_mae_table("trained model", compute_mae(model_preds, dish_trues))


if __name__ == "__main__":
    main()
