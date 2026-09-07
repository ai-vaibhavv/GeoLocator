"""Find confidence thresholds that actually predict a bad answer.

The first version of the caution used guessed cut-offs and stayed silent through
a 102 km miss. This measures which of the model's own signals correlate with
error on the test split, so the thresholds are chosen from data and the UI can
state the real effect rather than implying one.

    python dashboard/calibrate.py [n]
"""

import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ROOT)

from PIL import Image
import inference

n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
PREDICTOR = "anchor+snap+topk"

rows = list(csv.DictReader(open("splits/test.csv")))
random.seed(0)
rows = random.sample(rows, min(n, len(rows)))
inference.load()

recs = []
for i, r in enumerate(rows, 1):
    p = inference.locate(Image.open(f"geo_dataset/train/{r['filename']}"),
                         predictor=PREDICTOR)
    recs.append({
        "err": inference.haversine_km(p.lat, p.lng, float(r["lat"]), float(r["lng"])),
        "country_prob": p.country_prob,
        "cell_prob": p.cell_prob,
        "gate": p.gate,
    })
    if i % 100 == 0:
        print(f"  {i}/{len(rows)}", flush=True)


def median(xs):
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


overall = median([r["err"] for r in recs])
print(f"\noverall median {overall:.1f} km over {len(recs)} photos\n")

best = None
for sig in ("country_prob", "cell_prob", "gate"):
    print(f"{sig}:")
    for q in (0.10, 0.15, 0.20, 0.25, 0.30):
        vals = sorted(r[sig] for r in recs)
        cut = vals[int(q * len(vals))]
        flagged = [r["err"] for r in recs if r[sig] <= cut]
        rest = [r["err"] for r in recs if r[sig] > cut]
        if not flagged or not rest:
            continue
        mf, mr = median(flagged), median(rest)
        lift = mf / mr
        print(f"  <= {cut:.3f} (worst {q:.0%}): flagged median {mf:7.1f} km  "
              f"vs {mr:6.1f} km  lift x{lift:.2f}  n={len(flagged)}")
        if best is None or lift > best["lift"]:
            best = {"signal": sig, "cut": cut, "flagged_median": mf,
                    "rest_median": mr, "lift": lift, "n": len(flagged),
                    "coverage": len(flagged) / len(recs)}
    print()

print(f"best single signal: {best}")
out = {"predictor": PREDICTOR, "n": len(recs), "overall_median_km": overall,
       "best": best}
json.dump(out, open(os.path.join(HERE, "model_assets", "calibration.json"), "w"),
          indent=2)
print("wrote model_assets/calibration.json")
