import torch


def topk_anchor_prediction(
    pred_lat, pred_lng, p1, p2, g, A1, A2, c1lat, c1lng, c2lat, c2lng, k=3, snap=True
):
    """Re-estimate the anchor from the k most likely cells, keeping the offset.

    Args:
        pred_lat, pred_lng: the model's final prediction (anchor + offset).
        p1, p2: softmax probabilities over the coarse and fine partitions.
        g: per-image blend weight from the model's gate.
        A1, A2: cell -> country indicator matrices (see country_snap.py).
        c1lat, c1lng, c2lat, c2lng: the two partitions' centroid coordinates.
        k: how many cells to keep.
        snap: whether to mask to the most likely country first.

    Returns:
        (lat, lng, chosen): the re-anchored coordinates and the country index
        used, or None when snap is False.
    """
    a_lat = g * (p2 @ c2lat) + (1 - g) * (p1 @ c1lat)
    a_lng = g * (p2 @ c2lng) + (1 - g) * (p1 @ c1lng)

    gg = g.unsqueeze(1)
    W = torch.cat([(1 - gg) * p1, gg * p2], dim=1)
    slat = torch.cat([c1lat, c2lat])
    slng = torch.cat([c1lng, c2lng])

    chosen = None
    if snap:
        # Pick the most likely country over the joint mixture, then zero every cell outside it and renormalise.
        A = torch.cat([A1, A2], dim=0)
        chosen = (W @ A).argmax(1)
        m = (A[:, chosen].T > 0).to(W.dtype)
        q = W * m
        ssum = q.sum(1, keepdim=True)
        W = torch.where(ssum > 0, q / ssum.clamp(min=1e-12), W)

    # Keep the k highest-weight cells and renormalisee.
    k = min(int(k), W.shape[1])
    vals, idx = W.topk(k, dim=1)
    vals = vals / vals.sum(1, keepdim=True).clamp(min=1e-12)
    t_lat = (vals * slat[idx]).sum(1)
    t_lng = (vals * slng[idx]).sum(1)
    return t_lat + (pred_lat - a_lat), t_lng + (pred_lng - a_lng), chosen
