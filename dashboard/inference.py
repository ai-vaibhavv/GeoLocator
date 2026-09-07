"""Single-image geolocation for the dashboard.

Wraps src/ so the app only ever sees `locate(pil_image) -> Prediction`.
The checkpoint and the cell partitions are loaded once, on the first call,
and kept for the life of the process.
"""

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as cfg
from src.anchor_rule import topk_anchor_prediction
from src.backbone import DualGeoCellGeoNet
from src.country_snap import country_matrices, country_probs, snap_prediction
from src.dataset import MEAN, STD

# The inference rules the dashboard offers, each with the line of UI copy that
# explains it. "retrieval" only appears once model_assets/index.npz exists; see
# available_predictors() and prepare.py --images.
RETRIEVAL = "retrieval"

# key -> (nickname shown in the UI, blurb). The key is the protocol name the
# paper uses, and the dashboard prints it after the blurb, so a reader can line
# the dashboard up against the write-up.
PREDICTORS = {
    RETRIEVAL: (
        "Mitron",
        "Mitron, I have seen this street before. Goes through every photo the "
        "model was trained on, picks the one that looks most like yours, and "
        "reuses its location. Sharpest of the four, but it can only ever "
        "answer with a place it has already visited.",
    ),
    "anchor+snap+topk": (
        "Pappu",
        "Settles on a country, then narrows it down to the three best "
        "neighbourhoods and takes the middle. Gets there in the end, and more "
        "often than you would expect. Picked by default.",
    ),
    "anchor+snap": (
        "Muffler",
        "Wraps the whole country in one muffler and calls it a day. Names the "
        "country confidently, then averages across all of it. Rarely wildly "
        "wrong, rarely precise.",
    ),
    "anchor": (
        "Cockroach",
        "Scurries straight to an answer with no country check and no second "
        "guessing, just whatever the model says first. Useful for seeing how "
        "much the sanity checking is actually saving you.",
    ),
}

DEFAULT_PREDICTOR = "anchor+snap+topk"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Set GEO_HF_REPO to a Hugging Face model repo (e.g. "ai-vaibhavv/GeoLocator")
# to pull the weights from the Hub instead of shipping them in the app repo.
# Local files always win, so a checkout with weights present never hits the
# network and the Hub is only consulted for what is genuinely missing.
HF_REPO = os.environ.get("GEO_HF_REPO")


def _from_hub(filename):
    """Download one file from the configured Hub repo, or return None.

    Args:
        filename: the file's name in the repo.

    Returns:
        A local cached path, or None if no repo is configured or the fetch
        fails. Failing soft matters: a missing optional file (the retrieval
        index) should disable one feature, not take the app down.
    """
    if not HF_REPO:
        return None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=HF_REPO, filename=filename, token=os.environ.get("HF_TOKEN")
        )
    except Exception as exc:
        print(f"could not fetch {filename} from {HF_REPO}: {exc}", file=sys.stderr)
        return None


# Deployed, all three live beside this file; in the repo they sit at the root.
def _find(env, *candidates, hub_name=None):
    """First existing path among an env override, local candidates, then the Hub.

    Args:
        env: environment variable that overrides everything.
        candidates: local paths to try in order.
        hub_name: filename to fetch from GEO_HF_REPO when no local file exists.

    Returns:
        A path. It may not exist, so callers still check.
    """
    override = os.environ.get(env)
    if override:
        return override
    found = next((p for p in candidates if os.path.exists(p)), None)
    if found:
        return found
    if hub_name:
        pulled = _from_hub(hub_name)
        if pulled:
            return pulled
    return candidates[0]


CHECKPOINT = _find(
    "GEO_CHECKPOINT",
    os.path.join(HERE, "model.pt"),
    os.path.join(ROOT, "model.pt"),
    hub_name="model.pt",
)
# Not "assets/": Dash reserves that name for the files it serves statically.
CELLS = _find(
    "GEO_CELLS",
    os.path.join(HERE, "model_assets", "cells.npz"),
    os.path.join(HERE, "cells.npz"),
    hub_name="cells.npz",
)
INDEX = _find(
    "GEO_INDEX",
    os.path.join(HERE, "model_assets", "index.npz"),
    os.path.join(HERE, "index.npz"),
    hub_name="index.npz",
)
TRAIN_CSV = _find("GEO_TRAIN_CSV", os.path.join(ROOT, "splits", "train.csv"))

_STATE = None


def haversine_km(lat1, lng1, lat2, lng2, R=6371.0088):
    """Great-circle distance in km between two points in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class Prediction:
    """One image's answer, in the units the dashboard displays."""

    lat: float
    lng: float
    country: str
    country_prob: float
    cell_prob: float
    gate: float
    predictor: str
    predictor_label: str
    neighbour_km: float = None


def _cells():
    """Load the two partitions and their per-cell country labels.

    Returns:
        (c1lat, c1lng, c2lat, c2lng, cc, cc2) as numpy arrays.

    Raises:
        SystemExit: when neither the precomputed .npz nor the training csv the
            fallback refits from is available.
    """
    if os.path.exists(CELLS):
        z = np.load(CELLS, allow_pickle=False)
        return [z[k] for k in ("c1lat", "c1lng", "c2lat", "c2lng", "cc", "cc2")]

    if not os.path.exists(TRAIN_CSV):
        raise SystemExit(
            f"no cells at {CELLS} and no training csv at {TRAIN_CSV}; "
            f"run `python dashboard/prepare.py` first"
        )
    # Fallback: refit from the split. Only this path needs scikit-learn, so the
    # imports stay local and the deployed Space never pulls it in.
    from src.country_snap import cell_countries
    from src.geocells import fit_geocells
    from src.train import load_rows

    rows = load_rows(TRAIN_CSV)
    lats = [r["lat"] for r in rows]
    lngs = [r["lng"] for r in rows]
    _, c1lat, c1lng, _ = fit_geocells(lats, lngs, cfg.N_CELLS, seed=cfg.GEOCELL_SEED)
    _, c2lat, c2lng, _ = fit_geocells(
        lats, lngs, cfg.N_CELLS_FINE, seed=cfg.GEOCELL_SEED
    )
    return (
        np.asarray(c1lat),
        np.asarray(c1lng),
        np.asarray(c2lat),
        np.asarray(c2lng),
        cell_countries(c1lat, c1lng, rows),
        cell_countries(c2lat, c2lng, rows),
    )


def load(verbose=True):
    """Build the model, the country masks and the transform, once.

    Args:
        verbose: print a one-line summary of what was loaded.

    Returns:
        A dict holding the model, device, transform, snap tensors and the
        country name list the snapped indices point into.
    """
    global _STATE
    if _STATE is not None:
        return _STATE

    if not os.path.exists(CHECKPOINT):
        raise SystemExit(f"no such checkpoint: {CHECKPOINT} (set GEO_CHECKPOINT)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    c1lat, c1lng, c2lat, c2lng, cc, cc2 = _cells()

    # The checkpoint carries every weight, so the backbone starts from scratch
    # rather than re-reading the ImageNet .pth.
    model = DualGeoCellGeoNet(
        cfg.TIMM_ARCH,
        None,
        c1lat.tolist(),
        c1lng.tolist(),
        c2lat.tolist(),
        c2lng.tolist(),
        dropout=cfg.DROPOUT,
        target_size=cfg.TARGET_SIZE,
        task_weight_init_lambda=cfg.UNCERTAINTY_INIT_LAMBDA,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    A1, A2, countries = country_matrices(cc, cc2, device)
    centroids = [
        torch.tensor(a, dtype=torch.float32, device=device)
        for a in (c1lat, c1lng, c2lat, c2lng)
    ]

    features, coords = None, None
    if os.path.exists(INDEX):
        z = np.load(INDEX, allow_pickle=False)
        features = torch.tensor(z["features"], dtype=torch.float32, device=device)
        features = torch.nn.functional.normalize(features, dim=1)
        coords = torch.tensor(z["coords"], dtype=torch.float32, device=device)

    _STATE = {
        "model": model,
        "device": device,
        "snap": (A1, A2, *centroids),
        "countries": countries,
        "features": features,
        "coords": coords,
        "transform": T.Compose(
            [
                T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        ),
    }
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        index = f", {features.shape[0]:,}-row index" if features is not None else ""
        print(
            f"loaded {CHECKPOINT} on {device} "
            f"({n:,} params, {len(countries)} countries{index})"
        )
    return _STATE


def available_predictors():
    """The rules that can actually run here.

    Returns:
        A dict of predictor key -> (product name, blurb). "retrieval" is
        included only when the feature index has been built; see
        prepare.py --images.
    """
    st = load()
    return {
        k: v
        for k, v in PREDICTORS.items()
        if k != RETRIEVAL or st["features"] is not None
    }


def locate(image, predictor=DEFAULT_PREDICTOR, topk=None):
    """Predict where a single photo was taken.

    Args:
        image: a PIL image or an (H, W, 3) uint8 array.
        predictor: which inference rule to apply; one of PREDICTORS.
        topk: how many cells "anchor+snap+topk" keeps, or None for
            cfg.ANCHOR_TOPK. Ignored by the other rules.

    Returns:
        A Prediction.

    Raises:
        ValueError: on an unknown predictor.
    """
    if predictor not in PREDICTORS:
        raise ValueError(f"unknown predictor {predictor!r}; pick one of {list(PREDICTORS)}")

    st = load()
    if predictor == RETRIEVAL and st["features"] is None:
        raise ValueError(
            f"retrieval needs a feature index at {INDEX}; "
            f"build one with `python dashboard/prepare.py --images <train dir>`"
        )
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    x = st["transform"](image.convert("RGB")).unsqueeze(0).to(st["device"])

    with torch.no_grad():
        pred_lat, pred_lng, cell_logits, cell2_logits, g = st["model"](x)
        p1 = torch.softmax(cell_logits, dim=1)
        p2 = torch.softmax(cell2_logits, dim=1)
        A1, A2 = st["snap"][0], st["snap"][1]
        cp = country_probs(p1, p2, g, A1, A2)[0]

        neighbour_km = None
        if predictor == RETRIEVAL:
            # Cosine similarity, because both sides are unit-norm.
            q = torch.nn.functional.normalize(st["model"].last_feat, dim=1)
            sim = (q @ st["features"].T)[0]
            j = int(sim.argmax())
            lat = st["coords"][j, 0].reshape(1)
            lng = st["coords"][j, 1].reshape(1)
            chosen = None
            neighbour_km = float(haversine_km(
                float(lat[0]), float(lng[0]), float(pred_lat[0]), float(pred_lng[0])
            ))
        elif predictor == "anchor":
            # No snapping, so nothing picks a country; report the mixture's own.
            lat, lng, chosen = pred_lat, pred_lng, None
        elif predictor == "anchor+snap":
            lat, lng, chosen = snap_prediction(
                pred_lat, pred_lng, p1, p2, g, *st["snap"]
            )
        else:
            lat, lng, chosen = topk_anchor_prediction(
                pred_lat, pred_lng, p1, p2, g, *st["snap"],
                k=int(topk or cfg.ANCHOR_TOPK),
            )

    i = int(chosen[0]) if chosen is not None else int(cp.argmax())
    return Prediction(
        lat=float(np.clip(float(lat[0]), -90.0, 90.0)),
        lng=((float(lng[0]) + 180.0) % 360.0) - 180.0,
        country=str(st["countries"][i]),
        country_prob=float(cp[i]),
        cell_prob=float(p1[0].max()),
        gate=float(g[0]),
        predictor=predictor,
        predictor_label=PREDICTORS[predictor][0],
        neighbour_km=neighbour_km,
    )
