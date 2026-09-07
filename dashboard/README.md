# Dashboard

A Plotly Dash app wrapped around the trained geolocation model. Upload a photo,
get a coordinate, a country, and a pin on an OpenStreetMap map.

**Scope.** The training split is 12 European countries (Belarus, Finland,
France, Germany, Iceland, Italy, Norway, Poland, Spain, Sweden, Turkey, United
Kingdom) at roughly 700 photos each. There is no class for anywhere else, so a
photo from outside Europe returns one of these twelve regardless. The app states
this under the controls rather than letting the map imply global coverage.

## Run it locally

From the repository root, with the root `requirements.txt` already installed:

```
pip install -r dashboard/requirements.txt
python dashboard/app.py
```

That serves on <http://127.0.0.1:8050>. `--port` and `--host` override the
defaults, and `--debug` turns on hot reloading.

The app looks for `model.pt` and `splits/train.csv` at the repository root.
`GEO_CHECKPOINT`, `GEO_CELLS` and `GEO_TRAIN_CSV` override those paths.

The first run refits the two k-means geo-cell partitions from the training
split, which takes about ten seconds. Run `python dashboard/prepare.py` once to
cache them to `model_assets/cells.npz` and skip that on every later start.

## Picking a method

The dropdown switches between the inference rules and re-scores the photo you
already uploaded when you change it. The UI shows a nickname; the protocol name
underneath each one is what `src/` and the paper call it.

| Shown as | Protocol | What it does |
| --- | --- | --- |
| Mitron | `retrieval` | Finds the training photo that looks most like yours and reuses its location. Best median error, but it can only answer with a place the training set has seen. Needs the feature index, below. |
| Pappu | `anchor+snap+topk` | Decides on a country, then narrows to its three best cells. The default. |
| Muffler | `anchor+snap` | Decides on a country, then averages across all of its cells. |
| Cockroach | `anchor` | No country correction at all. |

Each option's blurb in the app ends with its protocol name in bold, so the
dashboard and the paper stay easy to line up.

### Enabling Mitron

`retrieval` is a 1-NN lookup over backbone features, so it needs an index built
from the labelled training images. Build it once and the deployed app never
needs the images themselves:

```
python dashboard/prepare.py --images geo_dataset/train
```

That writes `model_assets/index.npz`, about 32 MB of float16 vectors plus their
coordinates, and takes a few minutes on CPU. Without it the app simply does not
offer the option, and `build_space.py` says so as it skips the file.

## Accuracy, per protocol

The repo's `test_summary.json` reports the **head's** own metrics, which is the
`anchor` protocol alone (99.32 km median). It is not the whole model's score:
the default `retrieval` protocol reaches 37.06 km on the same split. Quoting one
number for all four methods would flatter the weakest and undersell the
strongest, so the dashboard measures each one:

```
python dashboard/bench.py
```

That scores every protocol over `splits/test.csv` in a single forward pass per
photo and writes `model_assets/protocol_metrics.json`. The app reads it to size
the ring drawn around the pin and to word the accuracy caption, both of which
then change with the selected method. Without the file the app still runs; the
caption says the numbers have not been measured, and the ring falls back to
99 km.

## Deploy to Hugging Face Spaces

Dash is not one of the native Space SDKs, so this deploys as a **Docker** Space.
The free **CPU basic** tier (2 vCPU, 16 GB RAM) is plenty: the checkpoint is
18 MB and inference is well under a second on CPU.

1. Cache the partitions and assemble the Space directory:

   ```
   python dashboard/prepare.py
   python dashboard/build_space.py
   ```

   That writes `dashboard/space/`, holding the app, `src/`, the `Dockerfile`,
   `model.pt`, `model_assets/cells.npz`, and `model_assets/index.npz` if you
   built it. That is everything the Space needs and nothing else. The training
   CSV and the training photos are never copied; only the 320 fitted cell
   centroids, their country labels, and (optionally) the feature vectors are.

2. Create a Space at <https://huggingface.co/new-space> with **SDK: Docker** and
   **Hardware: CPU basic (free)**.

3. Push `dashboard/space/` to it. `build_space.py` prints the exact commands.
   `model.pt` goes through Git LFS, since Spaces reject plain files over 10 MB.

The image builds in a few minutes. A free Space sleeps after about 48 hours of
no traffic and wakes up on the next visit.

## Where the weights live

Two options, both supported without code changes.

**Committed to the Space (default).** `build_space.py` copies `model.pt` and
`model_assets/` into `dashboard/space/`, and Git LFS carries them. Simplest, and
the Space is self-contained at about 50 MB.

**Pulled from a Hub model repo.** Set `GEO_HF_REPO` and the app fetches whatever
is missing locally from that repo at startup:

```
export GEO_HF_REPO=ai-vaibhavv/GeoLocator
python dashboard/app.py
```

Local files always take priority, so a checkout that already has the weights
never touches the network. A private repo needs `HF_TOKEN` as well. This keeps
the Space repo to a few hundred KB of code and versions the model separately.

To publish the weights (run these yourself; `hf` ships with `huggingface_hub`):

```
hf auth login
hf repo create GeoLocator --repo-type model
hf upload GeoLocator model.pt model.pt --repo-type model
hf upload GeoLocator dashboard/model_assets/cells.npz cells.npz --repo-type model
hf upload GeoLocator dashboard/model_assets/index.npz index.npz --repo-type model
```

Then add `GEO_HF_REPO` as a variable in the Space settings.

## Files

| File | Role |
| --- | --- |
| `app.py` | The Dash UI: layout, callbacks, and the Plotly map. |
| `theme.py` | Palette, fonts, and the page CSS. |
| `inference.py` | Loads the checkpoint once; `locate(image, predictor)`. |
| `prepare.py` | Fits the geo-cells into `model_assets/cells.npz`, and with `--images` builds the retrieval index. |
| `bench.py` | Scores every protocol on the test split into `model_assets/protocol_metrics.json`. |
| `build_space.py` | Assembles `dashboard/space/` for deployment. |
| `Dockerfile` | How the Space builds and serves the app. |
| `assets/` | Dash's static folder: the favicon in .ico, .png and apple-touch sizes. Model files live in `model_assets/` because Dash serves everything under `assets/`. |
| `requirements.txt` | The dashboard's extra dependencies for local runs. |
| `requirements-space.txt` | What the Space installs: CPU torch, no scikit-learn. |

## Design

Palette from [colorhunt.co](https://colorhunt.co/palette/3368a066a3bfc8dfdbf2efe7):
`#3368A0` deep blue, `#66A3BF` mid blue, `#C8DFDB` mint, `#F2EFE7` cream.
Type is Josefin Sans for headings, Inter for body text, IBM Plex Mono for
coordinates and figures.
