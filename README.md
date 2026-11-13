# Portion-Aware Calorie Estimation

Image regression model: upload a plate photo, get predicted calories/protein/fat/carb. Trained on a subset of [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (EfficientNet-B0 backbone, PyTorch/timm).

## Setup

```
pip install -r requirements.txt
```

## Data

Included in this repo:

- `data/metadata/dish_metadata_cafe1.csv` — calorie/macro labels per dish
- `data/splits/subset_train_ids.txt`, `subset_test_ids.txt` — the 850 / 150 dish split used for the final model
- `results/checkpoint.pt` — the trained model (16 MB), so evaluation and the demo run without retraining

**Not included (too large for git):** the extracted frames (877 dishes, 13,251 images, ~950 MB).
Download `nutrition5k_frames_subset.zip` here: **[[LINK]](https://drive.google.com/drive/folders/1h3rn0P0hyLZ0htkvrF4llVmBblxp-mJB?usp=drive_link)**, and unzip it into the project root:

```
unzip nutrition5k_frames_subset.zip   # creates data/videos/<dish_id>/frames_sampled5/*.jpeg
```

The zip contains only the frames used for training and evaluation, not the raw videos (8.6 GB), which are only needed to re-extract frames. All three commands below need the frames; the demo also uses them at startup to build its nearest-neighbor index.

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
