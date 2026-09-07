import csv
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from src.dataset import GeoDataset
from src.geocells import assign_geocells, fit_geocells
from src.backbone import DualGeoCellGeoNet


def haversine_km(lat1, lng1, lat2, lng2, R=6371.0088):
    """Great-circle distance in km between two batches of points.

    Args:
        lat1, lng1: first points, in degrees.
        lat2, lng2: second points, in degrees.
        R: Earth's mean radius in km.

    Returns:
        A tensor of distances in km, broadcast to the inputs' shape.
    """
    lat1, lng1, lat2, lng2 = (torch.deg2rad(x) for x in (lat1, lng1, lat2, lng2))
    d = (
        torch.sin((lat2 - lat1) / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lng2 - lng1) / 2) ** 2
    )
    return 2 * R * torch.asin(torch.sqrt(torch.clamp(d, 0, 1)))


def summarize(dists):
    """Aggregate per-image distances into the challenge's reported metrics.

    Args:
        dists: per-image haversine distances, in km.

    Returns:
        A dict with median_km, mean_km, frac_200km and frac_750km.
    """
    dists_sorted = sorted(dists)
    n = len(dists_sorted)
    mid = n // 2
    median = (
        dists_sorted[mid] if n % 2 else (dists_sorted[mid - 1] + dists_sorted[mid]) / 2
    )
    return {
        "median_km": median,
        "mean_km": sum(dists) / n,
        "frac_200km": sum(d < 200 for d in dists) / n,
        "frac_750km": sum(d < 750 for d in dists) / n,
    }


def load_rows(path):
    """Read a labelled CSV into row dicts, coercing the coordinates to floats.

    Args:
        path: path to a CSV of filename, country, lat, lng.

    Returns:
        A list of row dicts with lat and lng as floats.
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["lat"] = float(r["lat"])
        r["lng"] = float(r["lng"])
    return rows


def haversine_soft_targets(lat, lng, cell_lat, cell_lng, tau_km):
    """Build PIGEON-style (Haas et al. 2024) distance-smoothed cell targets.

    Args:
        lat, lng: true coordinates for the batch, in degrees.
        cell_lat, cell_lng: the partition's centroid coordinates.
        tau_km: smoothing bandwidth in km; larger means a softer target.

    Returns:
        A (batch, n_cells) tensor of target probabilities, softmax(-d_km / tau).
    """
    b, n = lat.shape[0], cell_lat.shape[0]
    d = haversine_km(
        lat[:, None].expand(b, n),
        lng[:, None].expand(b, n),
        cell_lat[None, :].expand(b, n),
        cell_lng[None, :].expand(b, n),
    )
    return torch.softmax(-d / tau_km, dim=1)


def soft_cross_entropy(logits, target):
    """Cross-entropy against a full target distribution rather than a class index.

    Args:
        logits: unnormalised (batch, n_classes) scores.
        target: (batch, n_classes) target probabilities.

    Returns:
        The mean cross-entropy over the batch, as a scalar tensor.
    """
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def evaluate(model, loader, device):
    """Score a loader, reporting the final prediction and its components.

    Args:
        model: the model to evaluate.
        loader: a loader yielding (image, lat, lng, cell, cell2).
        device: device to run the forward passes on.

    Returns:
        The summarize() dict plus both cell accuracies, the mean gate, the mean
        max cell probability, and the median error of each intermediate anchor
        (coarse soft, coarse hard, fine soft, and the blend without the offset).
    """
    model.eval()
    dists, cell_correct, cell2_correct, total = [], 0, 0, 0
    max_probs, soft_anchor_dists, hard_anchor_dists = [], [], []
    fine_anchor_dists, blend_anchor_dists, gates = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, lat, lng, cell, cell2 = [b.to(device) for b in batch]
            pred_lat, pred_lng, cell_logits, cell2_logits, g = model(imgs)

            def errors_km(plat, plng):
                return haversine_km(plat, plng, lat, lng).cpu().tolist()

            probs = torch.softmax(cell_logits, dim=1)
            probs2 = torch.softmax(cell2_logits, dim=1)
            soft_lat, soft_lng = probs @ model.cell_lat, probs @ model.cell_lng
            fine_lat, fine_lng = probs2 @ model.cell2_lat, probs2 @ model.cell2_lng
            top = cell_logits.argmax(dim=1)

            dists += errors_km(pred_lat, pred_lng)
            soft_anchor_dists += errors_km(soft_lat, soft_lng)
            hard_anchor_dists += errors_km(model.cell_lat[top], model.cell_lng[top])
            fine_anchor_dists += errors_km(fine_lat, fine_lng)
            blend_anchor_dists += errors_km(
                g * fine_lat + (1 - g) * soft_lat, g * fine_lng + (1 - g) * soft_lng
            )

            max_probs += probs.max(dim=1).values.cpu().tolist()
            gates += g.cpu().tolist()
            cell_correct += (top == cell).sum().item()
            cell2_correct += (cell2_logits.argmax(dim=1) == cell2).sum().item()
            total += cell.shape[0]
    summary = summarize(dists)
    summary["cell_acc"] = cell_correct / total
    summary["mean_max_cell_prob"] = sum(max_probs) / len(max_probs)
    summary["soft_anchor_median_km"] = summarize(soft_anchor_dists)["median_km"]
    summary["hard_anchor_median_km"] = summarize(hard_anchor_dists)["median_km"]
    summary["cell2_acc"] = cell2_correct / total
    summary["fine_anchor_median_km"] = summarize(fine_anchor_dists)["median_km"]
    summary["blend_anchor_median_km"] = summarize(blend_anchor_dists)["median_km"]
    summary["mean_gate"] = sum(gates) / len(gates)
    return summary


def train(cfg, train_csv, val_csv, test_csv=None, out_dir="outputs/run"):
    """Train the model on one train/val pair, keeping the best epoch on val.

    Args:
        cfg: config module supplying every hyperparameter; see src/config.py.
        train_csv: labelled rows to train on.
        val_csv: labelled rows to early-stop and select the checkpoint on.
        test_csv: labelled rows to score once at the end, or None.
        out_dir: directory for model.pt and test_summary.json.

    Returns:
        The best validation summary.
    """
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)

    train_rows = load_rows(train_csv)
    val_rows = load_rows(val_csv)
    test_rows = load_rows(test_csv) if test_csv else []

    train_lats = [r["lat"] for r in train_rows]
    train_lngs = [r["lng"] for r in train_rows]
    labels, cell_lat, cell_lng, km = fit_geocells(
        train_lats, train_lngs, cfg.N_CELLS, seed=cfg.GEOCELL_SEED
    )
    labels2, cell2_lat, cell2_lng, km2 = fit_geocells(
        train_lats, train_lngs, cfg.N_CELLS_FINE, seed=cfg.GEOCELL_SEED
    )
    for r, c, c2 in zip(train_rows, labels, labels2):
        r["cell"], r["cell2"] = c, c2
    for split in (val_rows, test_rows):
        if not split:
            continue
        lats = [r["lat"] for r in split]
        lngs = [r["lng"] for r in split]
        cells = assign_geocells(lats, lngs, km)
        cells2 = assign_geocells(lats, lngs, km2)
        for r, c, c2 in zip(split, cells, cells2):
            r["cell"], r["cell2"] = c, c2

    def loader_for(rows, augment):
        ds = GeoDataset(
            rows,
            cfg.IMG_DIR,
            cfg.IMAGE_SIZE,
            augment=augment,
            return_cell=True,
            return_cell2=True,
            aug_spec=getattr(cfg, "AUG_SPEC", None),
        )
        return DataLoader(
            ds, batch_size=cfg.BATCH_SIZE, shuffle=augment, num_workers=cfg.NUM_WORKERS
        )

    train_loader = loader_for(train_rows, cfg.AUGMENT)
    train_eval_loader = loader_for(train_rows, False)
    val_loader = loader_for(val_rows, False)
    test_loader = loader_for(test_rows, False) if test_rows else None

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.MAX_EPOCHS
    )

    os.makedirs(out_dir, exist_ok=True)

    best_val_median = float("inf")
    best_val_summary = None
    best_epoch = 0
    epochs_since_improvement = 0
    tau = cfg.LABEL_SMOOTH_KM

    for epoch in range(cfg.MAX_EPOCHS):
        epoch_start = time.time()
        model.train()
        for batch in train_loader:
            imgs, lat, lng = [b.to(device) for b in batch[:3]]
            optimizer.zero_grad()

            pred_lat, pred_lng, cell_logits, cell2_logits, _ = model(imgs)
            reg_loss = (
                haversine_km(pred_lat, pred_lng, lat, lng).mean() / cfg.REG_SCALE_KM
            )
            cell_loss = soft_cross_entropy(
                cell_logits,
                haversine_soft_targets(lat, lng, model.cell_lat, model.cell_lng, tau),
            )
            cell2_loss = soft_cross_entropy(
                cell2_logits,
                haversine_soft_targets(lat, lng, model.cell2_lat, model.cell2_lng, tau),
            )
            terms = [
                (cell_loss, model.log_var_cell),
                (cell2_loss, model.log_var_cell2),
                (reg_loss, model.log_var_reg),
            ]
            loss = sum(torch.exp(-lv) * L + lv for L, lv in terms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        train_summary = evaluate(model, train_eval_loader, device)
        val_summary = evaluate(model, val_loader, device)
        epoch_seconds = time.time() - epoch_start

        if val_summary["median_km"] < best_val_median - cfg.MIN_DELTA:
            best_val_median = val_summary["median_km"]
            best_val_summary = val_summary
            best_epoch = epoch + 1
            epochs_since_improvement = 0
            torch.save(model.state_dict(), f"{out_dir}/model.pt")
        else:
            epochs_since_improvement += 1

        print(
            f"epoch {epoch + 1}/{cfg.MAX_EPOCHS}  "
            f"train_median_km={train_summary['median_km']:.1f}  "
            f"val_median_km={val_summary['median_km']:.1f}  "
            f"cell_acc={val_summary['cell_acc']:.3f}  "
            f"cell2_acc={val_summary['cell2_acc']:.3f}  "
            f"gate={val_summary['mean_gate']:.3f}  "
            f"best={best_val_median:.1f}@{best_epoch}  "
            f"({epoch_seconds:.1f}s)"
        )

        if epochs_since_improvement >= cfg.PATIENCE:
            print(
                f"early stopping at epoch {epoch + 1} "
                f"(no improvement for {cfg.PATIENCE} epochs)"
            )
            break

    if test_loader is not None:
        model.load_state_dict(torch.load(f"{out_dir}/model.pt", map_location=device))
        test_summary = evaluate(model, test_loader, device)
        with open(f"{out_dir}/test_summary.json", "w") as f:
            json.dump(test_summary, f, indent=2)
        print(
            f"test_median_km={test_summary['median_km']:.1f}  "
            f"test_mean_km={test_summary['mean_km']:.1f}  "
            f"test_cell_acc={test_summary['cell_acc']:.3f}"
        )

    print(
        f"params={sum(p.numel() for p in model.parameters()):,}  "
        f"best_val_median_km={best_val_median:.1f}@{best_epoch}  "
        f"rows={len(train_rows):,}  out_dir={out_dir}"
    )
    return best_val_summary
