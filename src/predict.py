import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import GeoDataset
from src.geocells import fit_geocells
from src.backbone import DualGeoCellGeoNet
from src.anchor_rule import topk_anchor_prediction
from src.country_snap import cell_countries, country_matrices, snap_prediction
from src.train import load_rows

PREDICTORS = ["retrieval", "anchor", "anchor+snap", "anchor+snap+topk"]


def embed(loader, model, device):
    """Extract L2-normalised backbone features for a loader, in the loader's order.

    Both the index and the queries must go through this function, so that a dot
    product between two of its outputs is a cosine similarity.

    Args:
        loader: a loader over the images to embed.
        model: the model whose backbone provides the features.
        device: device to run the forward passes on.

    Returns:
        An (n, d) tensor of unit-norm feature vectors.
    """
    out = []
    with torch.no_grad():
        for batch in loader:
            model(batch[0].to(device))
            out.append(torch.nn.functional.normalize(model.last_feat, dim=1))
    return torch.cat(out)


def predict(loader, model, device, predictor, snap=None, topk=3):
    """Predict coordinates for every image in a loader, using the model's heads.

    Args:
        loader: a loader over the images to predict.
        model: the trained model.
        device: device to run the forward passes on.
        predictor: which anchor rule to apply.
        snap: the tensors the country mask needs, or None to skip snapping.
        topk: how many cells the "anchor+snap+topk" rule keeps.

    Returns:
        (lats, lngs, maxprobs, gates) as arrays in loader order; the last two
        are the labels-free collapse checks.
    """
    lats, lngs, maxprobs, gates = [], [], [], []
    with torch.no_grad():
        for imgs, _, _ in loader:
            pred_lat, pred_lng, cell_logits, cell2_logits, g = model(imgs.to(device))
            if snap is not None:
                p1 = torch.softmax(cell_logits, dim=1)
                p2 = torch.softmax(cell2_logits, dim=1)
                if predictor == "anchor+snap+topk":
                    pred_lat, pred_lng, _ = topk_anchor_prediction(
                        pred_lat, pred_lng, p1, p2, g, *snap, k=topk
                    )
                else:
                    pred_lat, pred_lng, _ = snap_prediction(
                        pred_lat, pred_lng, p1, p2, g, *snap
                    )
            lats += pred_lat.cpu().tolist()
            lngs += pred_lng.cpu().tolist()
            maxprobs += (
                torch.softmax(cell_logits, dim=1).max(dim=1).values.cpu().tolist()
            )
            gates += g.cpu().tolist()
    return np.array(lats), np.array(lngs), np.array(maxprobs), np.array(gates)


def predict_retrieval(loader, model, device, index_loader, index_rows):
    """Predict by 1-NN over the backbone's features against labelled images.

    Args:
        loader: a loader over the query images.
        model: the model whose backbone embeds both index and queries.
        device: device to run the forward passes on.
        index_loader: a loader over the index images, in index_rows order.
        index_rows: the labelled rows to index.

    Returns:
        (lats, lngs, maxprobs, gates), matching predict().
    """
    features = embed(index_loader, model, device)
    coords = torch.tensor(
        [[r["lat"], r["lng"]] for r in index_rows], dtype=torch.float32, device=device
    )
    print(f"index: {features.shape[0]:,} rows x {features.shape[1]}d")
    lats, lngs, maxprobs, gates = [], [], [], []
    with torch.no_grad():
        for imgs, _, _ in loader:
            _, _, cell_logits, _, g = model(imgs.to(device))
            q = torch.nn.functional.normalize(model.last_feat, dim=1)
            nearest = (q @ features.T).argmax(1)
            lats += coords[nearest, 0].cpu().tolist()
            lngs += coords[nearest, 1].cpu().tolist()
            maxprobs += (
                torch.softmax(cell_logits, dim=1).max(dim=1).values.cpu().tolist()
            )
            gates += g.cpu().tolist()
    return np.array(lats), np.array(lngs), np.array(maxprobs), np.array(gates)


def load_model(cfg, checkpoint, train_rows, device):
    """Rebuild the network around refitted cell partitions and load the weights.

    Args:
        cfg: config module supplying the architecture and partition settings.
        checkpoint: path to the model.pt to load.
        train_rows: the rows the checkpoint was trained on.
        device: device to place the model on.

    Returns:
        (model, cell_lat, cell_lng, cell2_lat, cell2_lng): the loaded model in
        eval mode and the two partitions' centroids.
    """
    lats = [r["lat"] for r in train_rows]
    lngs = [r["lng"] for r in train_rows]
    _, cell_lat, cell_lng, _ = fit_geocells(
        lats, lngs, cfg.N_CELLS, seed=cfg.GEOCELL_SEED
    )
    _, cell2_lat, cell2_lng, _ = fit_geocells(
        lats, lngs, cfg.N_CELLS_FINE, seed=cfg.GEOCELL_SEED
    )
    model = DualGeoCellGeoNet(
        cfg.TIMM_ARCH,
        cfg.PRETRAINED_PATH,
        cell_lat,
        cell_lng,
        cell2_lat,
        cell2_lng,
        dropout=cfg.DROPOUT,
        target_size=cfg.TARGET_SIZE,
        task_weight_init_lambda=cfg.UNCERTAINTY_INIT_LAMBDA,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    return model, cell_lat, cell_lng, cell2_lat, cell2_lng


def run(cfg, checkpoint, train_csv, out, images=None, labels=None, predictor=None):
    """Score a checkpoint and write a predictions csv, then print the checks.

    Args:
        cfg: config module supplying every setting; see src/config.py.
        checkpoint: path to the model.pt to score.
        train_csv: the csv the checkpoint was trained on; its rows refit the
            cell partitions and build the 1-NN index.
        out: path to write the predictions csv to.
        images: directory of unlabelled images to predict.
        labels: csv of labelled rows to predict instead of `images`.
        predictor: inference rule, or None to use cfg.PREDICTOR.

    Returns:
        None; the csv is written and the checks are printed.
    """
    if not os.path.exists(checkpoint):
        raise SystemExit(f"no such checkpoint: {checkpoint}")
    predictor = predictor or cfg.PREDICTOR
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = load_rows(train_csv)
    if labels:
        query_rows, query_dir = load_rows(labels), cfg.IMG_DIR
    else:
        query_rows = [
            {"filename": f, "lat": 0.0, "lng": 0.0}
            for f in sorted(os.listdir(images))
            if f.endswith(".jpg")
        ]
        query_dir = images
    query_loader = DataLoader(
        GeoDataset(query_rows, query_dir, cfg.IMAGE_SIZE, augment=False),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    model, cell_lat, cell_lng, cell2_lat, cell2_lng = load_model(
        cfg, checkpoint, train_rows, device
    )
    print(f"checkpoint {checkpoint}")

    if predictor == "retrieval":
        missing = [
            r["filename"]
            for r in train_rows[:64]
            if not os.path.exists(os.path.join(cfg.IMG_DIR, r["filename"]))
        ]
        if missing:
            print(
                f"training images missing under {cfg.IMG_DIR!r}, "
                f"falling back to 'anchor+snap+topk'"
            )
            predictor = "anchor+snap+topk"

    print(f"PREDICTOR = {predictor}")
    snap = None
    if predictor.startswith("anchor+snap"):
        cc = cell_countries(cell_lat, cell_lng, train_rows)
        cc2 = cell_countries(cell2_lat, cell2_lng, train_rows)
        A1, A2, _ = country_matrices(cc, cc2, device)
        centroids = [
            torch.tensor(a, dtype=torch.float32, device=device)
            for a in (cell_lat, cell_lng, cell2_lat, cell2_lng)
        ]
        snap = (A1, A2, *centroids)

    if predictor == "retrieval":
        index_loader = DataLoader(
            GeoDataset(train_rows, cfg.IMG_DIR, cfg.IMAGE_SIZE, augment=False),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
        )
        lats, lngs, maxprobs, gates = predict_retrieval(
            query_loader, model, device, index_loader, train_rows
        )
    else:
        lats, lngs, maxprobs, gates = predict(
            query_loader, model, device, predictor, snap, topk=cfg.ANCHOR_TOPK
        )

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "pred_lat", "pred_lng"])
        w.writerows(
            [r["filename"], f"{la:.6f}", f"{ln:.6f}"]
            for r, la, ln in zip(query_rows, lats, lngs)
        )

    print(f"\nwrote {out} ({len(query_rows)} rows)")


if __name__ == "__main__":
    from src import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", required=True, help="csv of the rows this checkpoint was trained on"
    )
    parser.add_argument("--checkpoint", default=f"{cfg.OUT_DIR}/model.pt")
    parser.add_argument(
        "--img-dir",
        default=cfg.IMG_DIR,
        help="directory holding the labelled images named by --train/--labels",
    )
    parser.add_argument(
        "--pretrained",
        default=cfg.PRETRAINED_PATH,
        help="imagenet weights for the backbone, or 'none' for none",
    )
    parser.add_argument(
        "--images",
        default="geo_dataset/holdout_public",
        help="directory of unlabelled images to predict",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="csv of labelled rows to predict instead of --images",
    )
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument(
        "--predictor",
        default=None,
        choices=PREDICTORS,
        help="overrides cfg.PREDICTOR for an ablation run",
    )
    args = parser.parse_args()

    cfg.IMG_DIR = args.img_dir
    cfg.PRETRAINED_PATH = (
        None if not args.pretrained or args.pretrained.lower() == "none"
        else args.pretrained
    )

    run(
        cfg,
        args.checkpoint,
        args.train,
        args.out,
        images=args.images,
        labels=args.labels,
        predictor=args.predictor,
    )
