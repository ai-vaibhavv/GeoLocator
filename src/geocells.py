import numpy as np
from sklearn.cluster import KMeans


def _to_xyz(lat_deg, lng_deg):
    """Project coordinates onto the unit sphere, so Euclidean distance is chordal.

    Args:
        lat_deg, lng_deg: coordinates in degrees.

    Returns:
        An (n, 3) array of unit-sphere (x, y, z) rows.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lng = np.radians(np.asarray(lng_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lng), cos_lat * np.sin(lng), np.sin(lat)], axis=1)


def _to_latlng(x, y, z):
    """Give an empty cluster a centroid by projecting its k-means centre back
    to degrees. Every cell index needs one, because the classifier has an
    output unit for each.

    Args:
        x, y, z: a point in the unit-sphere space _to_xyz maps into.

    Returns:
        (lat, lng) in degrees.
    """
    r = max(float(np.linalg.norm([x, y, z])), 1e-12)
    lat = float(np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0))))
    return lat, float(np.degrees(np.arctan2(y, x)))


def fit_geocells(lats, lngs, n_cells, seed=42):
    """Fit a k-means geo-cell partition to the given TRAINING coordinates.

    Args:
        lats, lngs: training coordinates in degrees.
        n_cells: number of cells; may not exceed the number of points.
        seed: k-means random_state.

    Returns:
        (labels, cell_lat, cell_lng, km): the per-row cell index, the two
        centroid lists indexed by cell, and the fitted KMeans for
        assign_geocells.
    """
    xyz = _to_xyz(lats, lngs)
    if n_cells > xyz.shape[0]:
        raise ValueError(f"n_cells={n_cells} exceeds {xyz.shape[0]} training points")
    km = KMeans(n_clusters=n_cells, random_state=seed, n_init=10).fit(xyz)
    labels = km.labels_

    lats_arr = np.asarray(lats, dtype=np.float64)
    lngs_arr = np.asarray(lngs, dtype=np.float64)
    cell_lat, cell_lng = [], []
    for k in range(n_cells):
        m = labels == k
        if m.any():
            cell_lat.append(float(lats_arr[m].mean()))
            cell_lng.append(float(lngs_arr[m].mean()))
        else:
            lat, lng = _to_latlng(*km.cluster_centers_[k])
            cell_lat.append(lat)
            cell_lng.append(lng)
    return labels.tolist(), cell_lat, cell_lng, km


def assign_geocells(lats, lngs, km):
    """Assign held-out rows to cells of an ALREADY-FITTED partition.

    Args:
        lats, lngs: coordinates in degrees.
        km: the KMeans returned by fit_geocells.

    Returns:
        A list of cell indices, one per row.
    """
    return km.predict(_to_xyz(lats, lngs)).tolist()
