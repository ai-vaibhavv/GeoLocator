import argparse
import csv
import math

parser = argparse.ArgumentParser()
parser.add_argument(
    "--predictions", required=True, help="csv with header filename,pred_lat,pred_lng"
)
parser.add_argument("--labels", default="geo_dataset/train_labels.csv")
args = parser.parse_args()


def haversine_km(lat1, lng1, lat2, lng2, R=6371.0088):
    """Great-circle distance in km between two points.

    Args:
        lat1, lng1: first point, in degrees.
        lat2, lng2: second point, in degrees.
        R: Earth's mean radius in km.

    Returns:
        The distance in km.
    """
    lat1, lng1, lat2, lng2 = map(math.radians, (lat1, lng1, lat2, lng2))
    d = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(d))


with open(args.labels) as f:
    truth = {
        r["filename"]: (float(r["lat"]), float(r["lng"])) for r in csv.DictReader(f)
    }

dists = []
with open(args.predictions) as f:
    for r in csv.DictReader(f):
        if r["filename"] in truth:
            lat, lng = truth[r["filename"]]
            dists.append(
                haversine_km(float(r["pred_lat"]), float(r["pred_lng"]), lat, lng)
            )

if not dists:
    raise SystemExit(f"no predicted filename appears in {args.labels}")

dists.sort()
n = len(dists)
mid = n // 2
median = dists[mid] if n % 2 else (dists[mid - 1] + dists[mid]) / 2
print(f"n: {n:,}")
print(f"median_km: {median:.2f}")
print(f"mean_km: {sum(dists) / n:.2f}")
print(f"frac < 200km: {sum(d < 200 for d in dists) / n:.4f}")
print(f"frac < 750km: {sum(d < 750 for d in dists) / n:.4f}")
