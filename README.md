# Portion-Aware Calorie Estimation

Image regression model: upload a plate photo, get predicted calories/protein/fat/carb. Trained on a subset of [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (EfficientNet-B0 backbone, PyTorch/timm).

## Setup

```
pip install -r requirements.txt
```

Data isn't included (see `data/`, gitignored) — download metadata/splits/videos from the Nutrition5k GCS bucket, then extract frames (every 5th) before running anything below.

## Usage

```
python -m src.train       # train the model, saves results/checkpoint.pt
python -m src.evaluate    # compare mean baseline, nearest-neighbor baseline, and the trained model
python -m app.app         # launch the Gradio demo (predictions + Grad-CAM + nearest-neighbor match)
```

## Layout

- `src/data.py` — metadata parsing, frame/label pairing, transforms, `FrameDataset`
- `src/model.py` — `CalorieRegressor` (EfficientNet-B0), `TargetNormalizer`
- `src/train.py` — two-phase training (frozen warmup, then fine-tune)
- `src/baselines.py` — mean and nearest-neighbor baselines
- `src/evaluate.py` — dish-level MAE, baseline comparison
- `src/gradcam.py` — shortcut-learning check (is the model looking at the food or the background?)
- `app/app.py` — demo UI
