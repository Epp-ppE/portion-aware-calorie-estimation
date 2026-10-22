import torch
from torch.utils.data import DataLoader

from src.data import TARGET_NAMES, FrameDataset, build_samples, build_transforms, load_dish_ids, parse_dish_metadata
from src.model import CalorieRegressor, TargetNormalizer

# Paths and hyperparameters as plain constants for now — this is still the
# smoke-test subset (132 dishes), so keep it simple and edit these directly
# rather than building a CLI before we know training even works.
METADATA_CSV = "data/metadata/dish_metadata_cafe1.csv"
FRAMES_ROOT = "data/videos"
TRAIN_SPLIT = "data/splits/subset_train_ids.txt"
TEST_SPLIT = "data/splits/subset_test_ids.txt"
CHECKPOINT_OUT = "results/checkpoint.pt"

BATCH_SIZE = 32
WARMUP_EPOCHS = 3
WARMUP_LR = 1e-3
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4


def run_epoch(model, loader, normalizer, optimizer, device):
    """One pass over the data. optimizer=None means eval mode (no learning)."""
    model.train() if optimizer is not None else model.eval()
    total_loss = 0.0
    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        targets_norm = normalizer.normalize(targets)

        with torch.set_grad_enabled(optimizer is not None): # no grad in eval mode
            preds = model(images)
            loss = torch.nn.functional.l1_loss(preds, targets_norm) #L1 loss = mean absolute error (MAE)

        if optimizer is not None:
            optimizer.zero_grad() #clear the gradients from the previous step
            loss.backward() #compute the gradients for the current step
            optimizer.step() #update weights based on the gradients and learning rate

        total_loss += loss.item() * images.size(0) #accumulate the loss for this batch, scaled by the number of samples in the batch
    return total_loss / len(loader.dataset)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    metadata = parse_dish_metadata(METADATA_CSV)
    train_ids = load_dish_ids(TRAIN_SPLIT)
    test_ids = load_dish_ids(TEST_SPLIT)
    train_samples = build_samples(FRAMES_ROOT, train_ids, metadata)
    test_samples = build_samples(FRAMES_ROOT, test_ids, metadata)
    print(f"train samples: {len(train_samples)}, test samples: {len(test_samples)}")

    train_set = FrameDataset(train_samples, build_transforms(train=True))
    test_set = FrameDataset(test_samples, build_transforms(train=False))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # normalizer stats come straight from train_samples' labels, no need to
    # load/transform any images just to compute mean/std.
    train_targets = torch.tensor(
        [[labels[name] for name in TARGET_NAMES] for _, labels, _ in train_samples], dtype=torch.float32
    )
    normalizer = TargetNormalizer.from_targets(train_targets)
    print("target means:", normalizer.means)
    print("target stds:", normalizer.stds)

    model = CalorieRegressor().to(device)

    # --- phase 1: warmup, backbone frozen ---
    model.set_backbone_frozen(True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=WARMUP_LR)
    for epoch in range(WARMUP_EPOCHS):
        train_loss = run_epoch(model, train_loader, normalizer, optimizer, device)
        print(f"[warmup {epoch}] train_loss={train_loss:.4f}")

    # --- phase 2: fine-tune, everything unfrozen ---
    model.set_backbone_frozen(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
    for epoch in range(FINETUNE_EPOCHS):
        train_loss = run_epoch(model, train_loader, normalizer, optimizer, device)
        test_loss = run_epoch(model, test_loader, normalizer, None, device)
        print(f"[finetune {epoch}] train_loss={train_loss:.4f} test_loss={test_loss:.4f}")

    torch.save(
        {"model_state": model.state_dict(), "means": normalizer.means, "stds": normalizer.stds},
        CHECKPOINT_OUT,
    )
    print("saved checkpoint to", CHECKPOINT_OUT)


if __name__ == "__main__":
    main()
