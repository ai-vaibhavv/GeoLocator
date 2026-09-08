# Dashboard

A Plotly Dash app wrapped around the trained geolocation model. Upload a photo,
get a coordinate, a country, and a pin on an OpenStreetMap map.

**Live: <https://24c16f08-880f-432f-8811-3667f58c689e.plotly.app/>**

**Scope.** The training split is 12 European countries (Belarus, Finland,
France, Germany, Iceland, Italy, Norway, Poland, Spain, Sweden, Turkey, United
Kingdom) at roughly 700 photos each. There is no class for anywhere else, so a
photo from outside Europe returns one of these twelve regardless. The app states
this under the controls rather than letting the map imply global coverage.

## Run it locally

The weights are not in the repository (`model.pt` and `model_assets/` are
gitignored), so a fresh clone needs them from the latest release:

```
gh release download --pattern 'model.pt' --dir .
gh release download --pattern '*.npz' --pattern '*.json' --dir dashboard/model_assets
```

`index.npz` is only needed for the Mitron method; without it the app runs and
hides that option. Then, with the root `requirements.txt` already installed:

```
pip install -r dashboard/requirements.txt
python dashboard/app.py
```

That serves on <http://127.0.0.1:8050>. `--port` and `--host` override the
defaults, and `--debug` turns on hot reloading.

The app looks for `model.pt` and `splits/train.csv` at the repository root.
`GEO_CHECKPOINT`, `GEO_CELLS`, `GEO_INDEX` and `GEO_TRAIN_CSV` override those
paths.

The first run refits the two k-means geo-cell partitions from the training
split, which takes about ten seconds. Run `python dashboard/prepare.py` once to
cache them to `model_assets/cells.npz` and skip that on every later start.

## Picking a method

The dropdown switches between the inference rules and re-scores the photo you
already uploaded when you change it. The UI shows a nickname; the protocol name
underneath each one is what `src/` and the paper call it.

| Shown as | Protocol | Median error | What it does |
| --- | --- | --- | --- |
| Mitron | `retrieval` | 37 km | Finds the training photo that looks most like yours and reuses its location. Sharpest, but can only answer with a place the training set has seen. Needs the feature index, below. |
| Pappu | `anchor+snap+topk` | 75 km | Decides on a country, then narrows to its three best cells. The default. |
| Muffler | `anchor+snap` | 82 km | Decides on a country, then averages across all of its cells. |
| Cockroach | `anchor` | 99 km | No country correction at all. |

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
offer the option, and `build_cloud.py` says so as it skips the file.

## Accuracy, per protocol

The repo's `test_summary.json` reports the **head's** own metrics, which is the
`anchor` protocol alone (99.32 km median). It is not the whole model's score:
the default `retrieval` protocol reaches 37 km on the same split. Quoting one
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

## Confidence warning

The app flags predictions its own signals say are shaky. Those thresholds are
measured, not guessed:

```
python dashboard/calibrate.py
```

Over 500 test photos, `country_prob <= 0.81` picks out photos with a median
error of 611 km against 42 km for the rest, and `cell_prob <= 0.51` behaves
similarly. The gate was tested too and runs the *other* way (low gate scores
better, 35 km vs 93 km), so it is deliberately not used as a warning signal.
Results land in `model_assets/calibration.json`.

## Deploy to Plotly Cloud

Plotly hosts Dash apps directly, and the Free plan covers one public app. No
Dockerfile, no Git LFS: assemble a folder and upload it.

```
python dashboard/build_cloud.py
```

That writes `dashboard/cloud/` at about 50 MB, inside the documented limit of
80 MiB for a direct upload (200 MiB when publishing from dev tools). Then either
upload the folder at <https://cloud.plotly.com>, or:

```
pip install "dash[cloud]"
python dashboard/cloud/app.py --debug
```

and use the dev tools panel: Plotly Cloud, Sign In, name the app, Publish.

**Make the app public** under its Settings, Sharing tab. A private app redirects
to a login page. Public apps are free; private apps and a custom (non-hashed)
URL both need a paid plan.

To keep the bundle to a few hundred KB, leave the weights out and pull them from
a Hugging Face model repo at startup:

```
python dashboard/build_cloud.py --hub <user>/GeoLocator
```

That writes `hf_repo.txt` into the bundle, so no environment variable is needed
on the host.

## Where the weights live

**Bundled (default).** `build_cloud.py` copies `model.pt` and `model_assets/`
into `dashboard/cloud/`. Self-contained, no network at startup.

**Pulled from a Hub model repo.** Set `GEO_HF_REPO`, or let `--hub` write
`hf_repo.txt`, and the app fetches whatever is missing locally from that repo at
startup. Local files always take priority, so a checkout that already has the
weights never touches the network. A private repo needs `HF_TOKEN` as well.

To publish the weights (`hf` ships with `huggingface_hub`):

```
hf auth login
hf repo create GeoLocator --repo-type model
hf upload GeoLocator model.pt model.pt --repo-type model
hf upload GeoLocator dashboard/model_assets/cells.npz cells.npz --repo-type model
hf upload GeoLocator dashboard/model_assets/index.npz index.npz --repo-type model
```

## Files

| File | Role |
| --- | --- |
| `app.py` | The Dash UI: layout, callbacks, and the Plotly map. |
| `theme.py` | Palette, fonts, and the page CSS. |
| `inference.py` | Loads the checkpoint once; `locate(image, predictor)`. |
| `prepare.py` | Fits the geo-cells into `model_assets/cells.npz`, and with `--images` builds the retrieval index. |
| `bench.py` | Scores every protocol on the test split into `model_assets/protocol_metrics.json`. |
| `calibrate.py` | Measures which confidence signals predict a bad answer. |
| `build_cloud.py` | Assembles `dashboard/cloud/` for Plotly Cloud. |
| `requirements.txt` | The dashboard's extra dependencies for local runs. |
| `requirements-cloud.txt` | What Plotly Cloud installs: CPU torch, no scikit-learn. |
| `assets/` | Dash's static folder: the favicon. Model files live in `model_assets/` because Dash serves everything under `assets/`. |

## Design

Palette from [colorhunt.co](https://colorhunt.co/palette/3368a066a3bfc8dfdbf2efe7):
`#3368A0` deep blue, `#66A3BF` mid blue, `#C8DFDB` mint, `#F2EFE7` cream.
Type is Josefin Sans for headings, Inter for body text, IBM Plex Mono for
coordinates and figures.
