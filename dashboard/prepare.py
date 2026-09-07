"""Precompute everything the dashboard needs that depends on the training split.

The two geo-cell partitions are k-means fits over the training coordinates, and
each cell's country is the country of its nearest training row. None of that
changes between runs, so it is fitted once here and written to
model_assets/cells.npz. The deployed Space then starts instantly, and never
ships the training CSV or depends on scikit-learn.

    python dashboard/prepare.py

Pass --images to also build the retrieval index. That embeds every training
photo through the backbone and stores the vectors, which is what lets the
"retrieval" rule run without shipping the images themselves. It needs the
labelled training images on disk and takes a few minutes on CPU.

    python dashboard/prepare.py --images geo_dataset/train
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as cfg
from src.country_snap import cell_countries
from src.geocells import fit_geocells
from src.train import load_rows

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(HERE, "model_assets")


def build_index(train_csv, images, out, batch_size=64):
    """Embed every training photo, and save the vectors with their coordinates.

    The retrieval rule is a 1-NN lookup over these vectors, so the deployed app
    needs the vectors but never the photos. Stored as float16: at 8k rows of
    2048-d features that is about 32 MB instead of 64, and the extra noise is
    far below the gap between neighbours.

    Args:
        train_csv: the csv naming the labelled images.
        images: directory the filenames in that csv are relative to.
        out: path of the .npz to write.
        batch_size: images per forward pass.

    Returns:
        None; the .npz is written.
    """
    import torch
    from torch.utils.data import DataLoader

    # One-off batch job, so let it have every core rather than torch's default.
    torch.set_num_threads(os.cpu_count() or 4)

    sys.path.insert(0, HERE)
    import inference
    from src.dataset import GeoDataset

    if not os.path.isdir(images):
        raise SystemExit(f"no such image directory: {images}")

    rows = load_rows(train_csv)
    missing = [r["filename"] for r in rows if not os.path.exists(os.path.join(images, r["filename"]))]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(rows)} images are not under {images!r}, "
            f"starting with {missing[0]}"
        )

    st = inference.load()
    model, device = st["model"], st["device"]
    loader = DataLoader(
        GeoDataset(rows, images, cfg.IMAGE_SIZE, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, (os.cpu_count() or 4)),
    )

    feats = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            model(batch[0].to(device))
            feats.append(
                torch.nn.functional.normalize(model.last_feat, dim=1).cpu()
            )
            done = min((i + 1) * batch_size, len(rows))
            print(f"\r  embedded {done:,}/{len(rows):,}", end="", flush=True)
    print()

    features = torch.cat(feats).numpy().astype(np.float16)
    coords = np.array([[r["lat"], r["lng"]] for r in rows], dtype=np.float32)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, features=features, coords=coords)
    print(
        f"wrote {out} ({os.path.getsize(out) / 1024 / 1024:.1f} MB): "
        f"{features.shape[0]:,} x {features.shape[1]}d"
    )


def main(train_csv, out):
    """Fit both partitions and label their cells, then save them.

    Args:
        train_csv: the csv the checkpoint was trained on.
        out: path of the .npz to write.

    Returns:
        None; the .npz is written.
    """
    rows = load_rows(train_csv)
    lats = [r["lat"] for r in rows]
    lngs = [r["lng"] for r in rows]
    print(f"{len(rows):,} training rows from {train_csv}")

    _, c1lat, c1lng, _ = fit_geocells(lats, lngs, cfg.N_CELLS, seed=cfg.GEOCELL_SEED)
    _, c2lat, c2lng, _ = fit_geocells(
        lats, lngs, cfg.N_CELLS_FINE, seed=cfg.GEOCELL_SEED
    )
    cc = cell_countries(c1lat, c1lng, rows)
    cc2 = cell_countries(c2lat, c2lng, rows)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(
        out,
        c1lat=np.asarray(c1lat, dtype=np.float64),
        c1lng=np.asarray(c1lng, dtype=np.float64),
        c2lat=np.asarray(c2lat, dtype=np.float64),
        c2lng=np.asarray(c2lng, dtype=np.float64),
        cc=cc,
        cc2=cc2,
    )
    n = len(sorted(set(cc) | set(cc2)))
    size = os.path.getsize(out) / 1024
    print(
        f"wrote {out} ({size:.1f} KB): {cfg.N_CELLS} + {cfg.N_CELLS_FINE} cells "
        f"over {n} countries"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(ROOT, "splits", "train.csv"))
    ap.add_argument("--out", default=os.path.join(ASSETS, "cells.npz"))
    ap.add_argument(
        "--images",
        default=None,
        help="directory of labelled training images; builds the retrieval index",
    )
    ap.add_argument("--index-out", default=os.path.join(ASSETS, "index.npz"))
    a = ap.parse_args()

    main(a.train, a.out)
    if a.images:
        print(f"\nbuilding the retrieval index from {a.images}")
        build_index(a.train, a.images, a.index_out)
