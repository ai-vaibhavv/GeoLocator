"""Measure each protocol's error on the test split, for the dashboard's captions.

The repo's test_summary.json reports only the `anchor` rule, so quoting it for
all four protocols would be wrong. This scores every protocol the dashboard
offers on the same split and writes model_assets/protocol_metrics.json, which
app.py reads to size the map's ring and word the accuracy caption.

    python dashboard/bench.py
"""

import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ROOT)
from PIL import Image
import inference

rows = list(csv.DictReader(open("splits/test.csv")))
inference.load()
rules = list(inference.available_predictors())
dists = {r: [] for r in rules}

t0 = time.time()
for n, r in enumerate(rows, 1):
    img = Image.open(f"geo_dataset/train/{r['filename']}")
    for rule in rules:
        p = inference.locate(img, predictor=rule)
        dists[rule].append(
            inference.haversine_km(p.lat, p.lng, float(r["lat"]), float(r["lng"]))
        )
    if n % 100 == 0:
        print(f"  {n}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

out = {}
for rule in rules:
    d = sorted(dists[rule])
    n = len(d)
    mid = n // 2
    out[rule] = {
        "median_km": d[mid] if n % 2 else (d[mid-1]+d[mid])/2,
        "frac_200km": sum(x < 200 for x in d)/n,
        "frac_750km": sum(x < 750 for x in d)/n,
        "n": n,
    }
print(json.dumps(out, indent=2))
path = os.path.join(HERE, "model_assets", "protocol_metrics.json")
json.dump(out, open(path, "w"), indent=2)
print(f"wrote {path}")
