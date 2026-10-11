import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TARGET_NAMES = ["calories", "protein", "fat", "carb"]


def parse_dish_metadata(csv_path):
    """Parse a Nutrition5k dish_metadata_cafeN.csv file.

    Each row is: dish_id, total_calories, total_mass, total_fat, total_carb,
    total_protein, num_ingrs, then a variable number of 7-field ingredient
    blocks we don't need. Returns dish_id -> {calories, mass, fat, carb, protein}.
    """
    dishes = {}
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            dish_id = row[0]
            dishes[dish_id] = {
                "calories": float(row[1]),
                "mass": float(row[2]),
                "fat": float(row[3]),
                "carb": float(row[4]),
                "protein": float(row[5]),
            }
    return dishes


def load_dish_ids(split_path):
    """Read a split file (one dish_id per line) into a list.

    utf-8-sig tolerates a UTF-8 byte-order-mark (e.g. from PowerShell's
    Out-File) while still reading plain UTF-8 files normally.
    """
    dish_ids = []
    with open(split_path, encoding="utf-8-sig") as f:
        for line in f:
            dish_id = line.strip()
            if dish_id: #in case id is empty this will skip it
                dish_ids.append(dish_id)
    return dish_ids


def build_samples(frames_root, dish_ids, metadata):
    """Pair every extracted frame with its dish's labels.

    frames_root: path to the folder containing <dish_id>/frames_sampled5/.
    dish_ids: which dishes to include (e.g. from load_dish_ids).
    metadata: dish_id -> labels dict (from parse_dish_metadata).

    Returns a flat list of (image_path, labels, dish_id) tuples, one per
    frame, so a dish with 15 extracted frames contributes 15 entries, not 1.
    dish_id is carried along so predictions can later be grouped back by
    dish for dish-level evaluation.
    """
    samples = []
    for dish_id in dish_ids:
        frames_dir = Path(frames_root) / dish_id / "frames_sampled5"
        if not frames_dir.is_dir():
            continue
        labels = metadata[dish_id]
        for image_path in sorted(frames_dir.glob("*.jpeg")):
            samples.append((image_path, labels, dish_id))
    return samples


def build_transforms(train):
    """Image preprocessing. 224x224, ImageNet normalization.

    Train only gets flip/rotation/color jitter — no random scaling or
    random-resized-crop anywhere, since apparent food size on the plate is
    the signal the model needs to learn, and those augmentations would
    distort it.
    """
    steps = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [
        transforms.ToTensor(), #converts raw pixels (0-255 integers) into 0.0-1.0 floats. That's the 0-1 range.
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # match the pixel stats the pretrained backbone was trained on (ImageNet)
    ]
    return transforms.Compose(steps)


class FrameDataset(Dataset):
    """PyTorch Dataset over (image_path, labels, dish_id) samples from build_samples."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, labels, dish_id = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor([labels[name] for name in TARGET_NAMES], dtype=torch.float32)
        return image, target, dish_id
