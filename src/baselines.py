from collections import defaultdict

import timm
import torch
import torch.nn.functional as F

from src.data import TARGET_NAMES


def mean_baseline_predict(train_mean, dish_ids):
    """Always predict the train-set mean for every dish.

    train_mean: tensor of shape (4,) in real units (e.g. normalizer.means).
    Returns dish_id -> train_mean, same prediction for every dish.
    """
    return {dish_id: train_mean.clone() for dish_id in dish_ids}


def build_embedder(device):
    """Frozen, ImageNet-pretrained EfficientNet-B0 with no classification
    head (num_classes=0), so it returns pooled feature vectors instead of
    class scores. Never fine-tuned on our calorie labels — same backbone
    family as the main model, so any improvement the trained model shows
    is attributable to learning from calorie labels, not a better backbone.
    """
    embedder = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    embedder.eval()
    return embedder.to(device)


def embed_frames(embedder, loader, device):
    """Run every frame in loader through embedder.

    Returns L2-normalized embeddings, real-unit targets, and dish_ids, all
    aligned by index (row i of each belongs to the same frame).
    """
    all_embeddings = []
    all_targets = []
    all_dish_ids = []
    with torch.no_grad():
        for images, targets, dish_ids in loader:
            images = images.to(device)
            embeddings = embedder(images)
            embeddings = F.normalize(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu())
            all_targets.append(targets)
            all_dish_ids.extend(dish_ids)
    return torch.cat(all_embeddings), torch.cat(all_targets), all_dish_ids


def nearest_neighbor_predict(train_embeddings, train_targets, test_embeddings):
    """For each test embedding, copy the labels of the most similar train
    embedding (cosine similarity). Embeddings must already be L2-normalized,
    so a plain dot product equals cosine similarity.
    """
    similarity = test_embeddings @ train_embeddings.T
    nearest_idx = similarity.argmax(dim=1)
    return train_targets[nearest_idx]


def aggregate_dish_level(frame_preds, dish_ids):
    """Average per-frame predictions into one prediction per dish, same
    aggregation rule used for the main model's evaluation."""
    sums = defaultdict(lambda: torch.zeros(len(TARGET_NAMES)))
    counts = defaultdict(int)
    for pred, dish_id in zip(frame_preds, dish_ids):
        sums[dish_id] += pred
        counts[dish_id] += 1
    return {dish_id: sums[dish_id] / counts[dish_id] for dish_id in sums}
