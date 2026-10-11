"""Demo: upload a plate photo, see predicted calories/protein/fat/carb.
Run with: python -m app.app
"""

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.baselines import build_embedder, embed_frames
from src.data import TARGET_NAMES, FrameDataset, build_samples, build_transforms, load_dish_ids, parse_dish_metadata
from src.gradcam import GradCAM, overlay_heatmap
from src.model import CalorieRegressor, TargetNormalizer
from src.train import CHECKPOINT_OUT, FRAMES_ROOT, METADATA_CSV, TRAIN_SPLIT

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = build_transforms(train=False)

checkpoint = torch.load(CHECKPOINT_OUT, map_location=device, weights_only=True)
model = CalorieRegressor(pretrained=False).to(device)  # about to load our own trained weights below anyway
model.load_state_dict(checkpoint["model_state"])
model.eval()
normalizer = TargetNormalizer(checkpoint["means"], checkpoint["stds"])
gradcam = GradCAM(model)

# build the nearest-neighbor lookup index once at startup, not per-request
print("embedding training set for nearest-neighbor lookup...")
embedder = build_embedder(device)
metadata = parse_dish_metadata(METADATA_CSV)
train_ids = load_dish_ids(TRAIN_SPLIT)
train_samples = build_samples(FRAMES_ROOT, train_ids, metadata)
train_image_paths = [path for path, _, _ in train_samples]
train_eval_loader = DataLoader(FrameDataset(train_samples, build_transforms(train=False)), batch_size=32, shuffle=False)
train_embeddings, train_targets, train_dish_ids = embed_frames(embedder, train_eval_loader, device)
print(f"ready — {len(train_image_paths)} training frames indexed")


def predict(image):
    if image is None:
        return 0.0, 0.0, 0.0, 0.0, None, None, ""
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = normalizer.denormalize(model(tensor))[0].cpu()
    calories, protein, fat, carb = (float(v) for v in pred)

    # separate forward+backward pass (Grad-CAM needs gradients, so this one
    # can't run under torch.no_grad() like the prediction above)
    cam = gradcam.generate(tensor, target_index=0)  # 0 = explain the calorie prediction
    heatmap_image = overlay_heatmap(image, cam)

    # nearest-neighbor baseline, applied to just this one uploaded photo:
    # find the most visually similar training frame and show what it
    # actually was, so the model's prediction can be sanity-checked against
    # "what if we'd just copied the closest match instead"
    with torch.no_grad():
        query_embedding = F.normalize(embedder(tensor), dim=1).cpu()
    similarity = query_embedding @ train_embeddings.T
    nearest_idx = similarity.argmax(dim=1).item()

    nearest_image = Image.open(train_image_paths[nearest_idx]).convert("RGB")
    nn_cal, nn_protein, nn_fat, nn_carb = (float(v) for v in train_targets[nearest_idx])
    nearest_text = (
        f"Nearest training match ({train_dish_ids[nearest_idx]}) — actual values: "
        f"{nn_cal:.0f} kcal, {nn_protein:.1f}g protein, {nn_fat:.1f}g fat, {nn_carb:.1f}g carb"
    )

    return calories, protein, fat, carb, heatmap_image, nearest_image, nearest_text


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Plate photo"),
    outputs=[
        gr.Number(label="Calories (kcal)"),
        gr.Number(label="Protein (g)"),
        gr.Number(label="Fat (g)"),
        gr.Number(label="Carb (g)"),
        gr.Image(label="Grad-CAM — where the model looked for the calorie prediction (red = most responsible)"),
        gr.Image(label="Nearest-neighbor match (closest training photo)"),
        gr.Textbox(label="Nearest-neighbor match details"),
    ],
    title="Portion-Aware Calorie Estimation",
    description=(
        "Trained on a 132-dish smoke-test subset of Nutrition5k — still early-stage, "
        "currently underperforms the nearest-neighbor baseline. The Grad-CAM heatmap is "
        "an early diagnostic too, not a polished result — see extra/PROGRESS.md."
    ),
)

if __name__ == "__main__":
    demo.launch()
