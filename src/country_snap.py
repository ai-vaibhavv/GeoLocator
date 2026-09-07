import numpy as np
import torch

R_KM = 6371.0088


def _hav_np(lat1, lng1, lat2, lng2):
    """Great-circle distance in km between two points.

    Args:
        lat1, lng1: first point, in degrees.
        lat2, lng2: second points, in degrees; broadcast against the first.

    Returns:
        An array of distances in km.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lng2 - lng1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cell_countries(cell_lat, cell_lng, train_rows):
    """Label each cell with the country of its nearest TRAINING row.

    Args:
        cell_lat, cell_lng: cell centroid coordinates in degrees.
        train_rows: labelled rows carrying lat, lng and country.

    Returns:
        An array of country names, one per cell.
    """
    tlat = np.array([r["lat"] for r in train_rows])
    tlng = np.array([r["lng"] for r in train_rows])
    tc = np.array([r["country"] for r in train_rows])
    return np.array(
        [
            tc[np.argmin(_hav_np(la, lo, tlat, tlng))]
            for la, lo in zip(cell_lat, cell_lng)
        ]
    )


def country_matrices(cc, cc2, device):
    """Build one-hot cell -> country indicator matrices for both partitions.

    Args:
        cc, cc2: per-cell country names for the coarse and fine partitions.
        device: torch device to place the matrices on.

    Returns:
        (A1, A2, countries): the two indicator matrices and the sorted country
        list their columns are indexed by.
    """
    countries = sorted(set(cc) | set(cc2))
    idx = {c: i for i, c in enumerate(countries)}
    A1 = torch.zeros(len(cc), len(countries), device=device)
    for k, c in enumerate(cc):
        A1[k, idx[c]] = 1.0
    A2 = torch.zeros(len(cc2), len(countries), device=device)
    for k, c in enumerate(cc2):
        A2[k, idx[c]] = 1.0
    return A1, A2, countries


def country_probs(p1, p2, g, A1, A2):
    """Blend the two cell distributions into one distribution over countries.

    Args:
        p1, p2: softmax probabilities over the coarse and fine partitions.
        g: per-image blend weight from the model's gate.
        A1, A2: cell -> country indicator matrices.

    Returns:
        A (batch, n_countries) tensor of country probabilities.
    """
    gg = g.unsqueeze(1)
    return gg * (p2 @ A2) + (1 - gg) * (p1 @ A1)


def snap_anchor(p1, p2, g, A1, A2, c1lat, c1lng, c2lat, c2lng, chosen=None):
    """Recompute the anchor with both cell distributions masked to one country.

    Args:
        p1, p2: softmax probabilities over the coarse and fine partitions.
        g: per-image blend weight from the model's gate.
        A1, A2: cell -> country indicator matrices.
        c1lat, c1lng, c2lat, c2lng: the two partitions' centroid coordinates.
        chosen: country index per image; defaults to the model's own argmax.

    Returns:
        ((lat, lng), chosen): the masked anchor and the country it used.
    """
    if chosen is None:
        chosen = country_probs(p1, p2, g, A1, A2).argmax(1)
    m1 = (A1[:, chosen].T > 0).to(p1.dtype)
    m2 = (A2[:, chosen].T > 0).to(p2.dtype)
    q1, q2 = p1 * m1, p2 * m2
    s1, s2 = q1.sum(1, keepdim=True), q2.sum(1, keepdim=True)
    q1 = torch.where(s1 > 0, q1 / s1.clamp(min=1e-12), p1)
    q2 = torch.where(s2 > 0, q2 / s2.clamp(min=1e-12), p2)
    return (
        g * (q2 @ c2lat) + (1 - g) * (q1 @ c1lat),
        g * (q2 @ c2lng) + (1 - g) * (q1 @ c1lng),
    ), chosen


def snap_prediction(pred_lat, pred_lng, p1, p2, g, A1, A2, c1lat, c1lng, c2lat, c2lng):
    """Apply the full snap rule: recover the offset, re-anchor, re-add.

    Args:
        pred_lat, pred_lng: the model's final prediction (anchor + offset).
        p1, p2: softmax probabilities over the coarse and fine partitions.
        g: per-image blend weight from the model's gate.
        A1, A2: cell -> country indicator matrices.
        c1lat, c1lng, c2lat, c2lng: the two partitions' centroid coordinates.

    Returns:
        (lat, lng, chosen): the snapped coordinates and the country index used.
    """
    a_lat = g * (p2 @ c2lat) + (1 - g) * (p1 @ c1lat)
    a_lng = g * (p2 @ c2lng) + (1 - g) * (p1 @ c1lng)
    (s_lat, s_lng), chosen = snap_anchor(p1, p2, g, A1, A2, c1lat, c1lng, c2lat, c2lng)
    return s_lat + (pred_lat - a_lat), s_lng + (pred_lng - a_lng), chosen
