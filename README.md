# Where in Europe was this photo taken?

For an input image (512x512 street-level photo), predict lat/long where the photo was taken, rated by the great-circle (haversine) distance.


A **HGNetV2-B0** backbone serves a dual geo-cell classification head (64 coarse, 256 fine cells which are blended using a learned per image gate) and a zero-initialized coordinate offset head. At inference time the head is ignored-the learned feature from the backbone, used in 1-nearest-neighbour retrieval against the labeled training images, is what produces the prediction and the classifier provides training supervision.

| | |
|:--|:--|
| Parameters | **4,518,887** of 5,000,000 |
| Backbone | HGNetV2-B0, ImageNet-1k pretrained |
| Input | 512×512, the dataset's native resolution |
| Default inference | 1-NN retrieval over backbone features |
| Trained on | 8,230 labelled images, 92 epochs, ≈1 h 54 min on one NVIDIA A40 |
| Test median error | **37.06 km** |
| Live demo | **<https://24c16f08-880f-432f-8811-3667f58c689e.plotly.app/>** |

---

## Try it

An interactive dashboard is deployed at
<https://24c16f08-880f-432f-8811-3667f58c689e.plotly.app/>: upload a street
level photo and it returns a coordinate, the most likely country, and a pin on
a map, with a selector for all four inference protocols. Source and deployment
notes are in [`dashboard/`](dashboard/README.md).

---

## Getting Started

### Prerequisites

Python 3.12 and GPU.

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` locks down the exact versions the results were obtained with. The torch/torchvision pins correspond to CUDA 12.8 builds; install CPU-only wheel builds of the same versions on your CPU-only machine.

### Dataset

Place or symlink it at the package root:

```
geo_dataset/
├── train/              # 11,758 labelled images
├── train_labels.csv    # filename, country, iso, lat, lng
└── holdout_public/     # 2,400 unlabelled test images
```

### Pretrained backbone weights

`hgnetv2_b0.pth`  is shipped in this package, at the root where you expect `--pretrained` looking by default so that offline training runs without an additional step.

> **Note:** nothing in this code downloads anything. `src/backbone.py` calls `timm.create_model(arch, pretrained=False)` and loads the state-dict from disk itself. A missing file raises `FileNotFoundError`; it will not fetch, and will not silently fall back to random initialisation.

| | |
|:--|:--|
| Size | 15.2 MB, 315 tensors, 3,959,864 params |
| Source | `timm.create_model("hgnetv2_b0", pretrained=True)`, ImageNet-1k |

---

## Usage

Regenerates `predictions.csv` from the shipped `model.pt`. No training required.

```bash
python -m src.predict \
    --train      splits/train.csv \
    --checkpoint model.pt \
    --pretrained none \
    --img-dir    geo_dataset/train \
    --images     geo_dataset/holdout_public \
    --out        predictions.csv
```

### Train

```bash
python main.py --train splits/train.csv \
               --val   splits/val.csv \
               --test  splits/test.csv
```

Trains, then predicts the holdout images and writes `predictions.csv`.

| Flag | Default | Description |
|:--|:--|:--|
| `--train` | *required* | labelled rows to fit on |
| `--val` | *required* | drives early stopping and checkpoint selection |
| `--test` | `None` | optional; scored once at the end |
| `--img-dir` | `geo_dataset/train` | labelled training images |
| `--images` | `geo_dataset/holdout_public` | images to predict |
| `--pretrained` | `hgnetv2_b0.pth` | ImageNet weights, or `none` for random init |
| `--out` | `outputs/run` | output directory for `model.pt` |
| `--predictions` | `predictions.csv` | output CSV |
| `--seed` | `42` | optimisation seed only |

Runs to `MAX_EPOCHS` (100) with early stopping on validation median km after `PATIENCE` (10) epochs without improvement and outputs go under `--out`.

> **Training does not report the retrieval score.** The `test_summary.json` it writes contains only the **head's** own metrics i.e. the `anchor` rule. On the shipped model that file reads 99.32 km, while the default `retrieval` rule scores 37.06 km on the same split.

> To get the retrieval number for a checkpoint you have just trained, run the [evaluate](#evaluate) step against it:
>
> ```bash
> python -m src.predict --train splits/train.csv \
>     --checkpoint outputs/run/model.pt --pretrained none \
>     --labels splits/test.csv --out preds_test.csv
> python -m src.evaluate --predictions preds_test.csv --labels splits/test.csv
> ```

> The `predictions.csv` that `main.py` writes at the end *does* use retrieval it is only the scoring in `test_summary.json` that does not.

### Predict

Select the inference rule with `--predictor`; the default is `retrieval`.

| Rule | How it predicts | Needs `--img-dir` |
|:--|:--|:-:|
| `retrieval` *(default)* | 1-NN over backbone features against the labelled images | **yes** |
| `anchor` | gated 64/256 centroid blend + offset | no |
| `anchor+snap` | `anchor`, snapped to the predicted country | no |
| `anchor+snap+topk` | `anchor+snap` over the top-*k* cells rather than the argmax | no |

Output is one row per `.jpg` in `--images`, with header `filename,pred_lat,pred_lng`. If the training images are unreadable, retrieval cannot build its index and the script warns and falls back to `anchor+snap+topk` rather than crashing.

### Evaluate

```bash
python -m src.predict --train splits/train.csv --checkpoint model.pt \
    --pretrained none --labels splits/val.csv --out preds_val.csv

python -m src.evaluate --predictions preds_val.csv --labels splits/val.csv
```

---

## Project Structure

```
main.py                  entry point - trains, then writes predictions.csv
src/
├── config.py            every hyperparameter, in one file
├── backbone.py          the network definition
├── dataset.py           image loading and augmentation
├── geocells.py          k-means geo-cell partitions
├── train.py             training loop, losses, per-epoch evaluation
├── predict.py           inference -> predictions.csv
├── evaluate.py          scores a predictions file against labels
├── anchor_rule.py       top-k anchor inference rule
└── country_snap.py      country-snapping inference rule
splits/                  train.csv (8,230), val.csv (1,764), test.csv (1,764)
dashboard/               the deployed Dash app; see dashboard/README.md

hgnetv2_b0.pth           ImageNet-1k backbone weights (not in git; see releases)
model.pt                 shipped checkpoint, 4,518,887 params, epoch 82
                         (not in git; attached to the latest release)
predictions.csv          2,400 rows
test_summary.json        head-only metrics on splits/test.csv
```

### Module reference

| File | Contents |
|:--|:--|
| **`main.py`** | Argument parsing and the end-to-end pipeline: calls `train.train()`, then `predict.run()`. Validates both image directories *before* training starts, so a wrong path fails in seconds rather than after an epoch. |
| **`src/config.py`** | Flat module of constants: paths, architecture, optimiser, schedule, seeds, geo-cell counts, default predictor. |
| **`src/backbone.py`** | `DualGeoCellGeoNet` the timm backbone, a zero-init `detail_conv` residual applied before it, the shared `cell_stem` projection feeding both the 64- and 256-cell classifiers, the blend `gate`, and the zero-init `offset_head`. `forward()` returns the 5-tuple `(lat, lng, cell_logits, cell2_logits, gate)`. |
| **`src/dataset.py`** | `GeoDataset` maps a row dict to `(image, lat, lng[, cell][, cell2])`. Builds the transform pipeline from `AUG_DEFAULT` overridden by `cfg.AUG_SPEC`|
| **`src/geocells.py`** | `fit_geocells()` runs k-means on unit-sphere coordinates and returns per-row labels, the centroid lists, and the fitted `KMeans`; `assign_geocells()` maps held-out rows onto that already-fitted partition. Empty clusters still get a centroid, since the classifier has an output unit per cell. |
| **`src/train.py`** | The training loop, plus the metric and loss helpers: `haversine_km()`, `summarize()`, `load_rows()`, `haversine_soft_targets()` (distance-smoothed cell targets), `soft_cross_entropy()`, and `evaluate()`. `train()` orchestrates the run and writes `model.pt` / `test_summary.json`. |
| **`src/predict.py`** | `embed()` produces L2-normalised features so a dot product is a cosine; `predict_retrieval()` implements 1-NN against the index; `predict()` implements the three `anchor*` rules; `load_model()` rebuilds the network and loads a checkpoint; `run()` is the CLI entry point. |
| **`src/evaluate.py`** | Reads a predictions CSV plus a labels CSV and prints median/mean km and the `<200 km` / `<750 km` fractions. |
| **`src/anchor_rule.py`** | `topk_anchor_prediction()` averages over the top-*k* cells instead of the argmax, weighted by probability. |
| **`src/country_snap.py`** | `cell_countries()` labels each cell with its majority country, `country_matrices()` builds the cell -> country incidence matrices, and `snap_prediction()` pulls a prediction back into the most probable country's cells.|

---

## Configuration

All hyperparameters are in `src/config.py`

---

For Results and more details see [report](#report.pdf)