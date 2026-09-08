"""Assemble a folder ready to upload to Plotly Cloud.

Plotly Cloud hosts Dash apps directly, so unlike the Space build there is no
Dockerfile and no Git LFS: you upload a folder and it detects the entry point.
The only hard constraint is size, from dash.plotly.com/deployment:

    "The total size of your app files must be less than 200 MiB when publishing
    from dev tools or less than 80 MiB when uploading directly."

The default bundle carries the weights and lands around 51 MB, which clears
both. Pass --hub to leave the two large files out and fetch them from a Hugging
Face model repo at startup instead, which drops the bundle to a few hundred KB.

    python dashboard/build_cloud.py
    python dashboard/build_cloud.py --hub ai-vaibhavv/GeoLocator
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Destination name -> source path.
FILES = {
    "app.py": os.path.join(HERE, "app.py"),
    "inference.py": os.path.join(HERE, "inference.py"),
    "theme.py": os.path.join(HERE, "theme.py"),
    "requirements.txt": os.path.join(HERE, "requirements-cloud.txt"),
    "assets/favicon.ico": os.path.join(HERE, "assets", "favicon.ico"),
    "assets/favicon.png": os.path.join(HERE, "assets", "favicon.png"),
    "assets/apple-touch-icon.png": os.path.join(
        HERE, "assets", "apple-touch-icon.png"
    ),
    # Small enough to always ship: without them the app cannot start at all.
    "model_assets/cells.npz": os.path.join(HERE, "model_assets", "cells.npz"),
}

# Shipped when present; the app degrades gracefully rather than failing.
OPTIONAL = {
    "model_assets/protocol_metrics.json": os.path.join(
        HERE, "model_assets", "protocol_metrics.json"
    ),
}

# Left out by --hub and pulled from the Hub instead.
HEAVY = {
    "model.pt": os.path.join(ROOT, "model.pt"),
    "model_assets/index.npz": os.path.join(HERE, "model_assets", "index.npz"),
}


def main(out, hub):
    """Assemble the upload folder.

    Args:
        out: directory to assemble into; replaced if it exists.
        hub: a Hugging Face model repo id to fetch the weights from, or None to
            bundle them.

    Returns:
        None; prints a manifest and the publish steps.
    """
    wanted = dict(FILES)
    if not hub:
        wanted.update(HEAVY)

    missing = [s for s in wanted.values() if not os.path.exists(s)]
    if missing:
        for m in missing:
            print(f"missing: {m}", file=sys.stderr)
        raise SystemExit(
            "run `python dashboard/prepare.py` first, and check model.pt is present"
        )

    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    for name, src in {**wanted, **OPTIONAL}.items():
        if not os.path.exists(src):
            print(f"skipping {name} (not built)")
            continue
        dst = os.path.join(out, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    shutil.copytree(
        os.path.join(ROOT, "src"),
        os.path.join(out, "src"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    if hub:
        # Not every host lets you set environment variables, so the repo id
        # travels with the bundle.
        with open(os.path.join(out, "hf_repo.txt"), "w") as f:
            f.write(hub + "\n")

    total = 0
    print(f"\n{out}")
    for dirpath, dirnames, filenames in os.walk(out):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in sorted(filenames):
            p = os.path.join(dirpath, f)
            size = os.path.getsize(p)
            total += size
            print(f"  {os.path.relpath(p, out):<38} {size / 1024:>9,.1f} KB")
    mb = total / 1024 / 1024
    print(f"  {'total':<38} {mb:>9,.1f} MB")

    cap = "80 MiB direct upload / 200 MiB from dev tools"
    print(f"\n  limit: {cap} -> {'OK' if mb < 80 else 'TOO BIG, use --hub'}")
    if hub:
        print(f"  weights come from https://huggingface.co/{hub} at startup")

    print(
        "\nto publish:\n"
        "  pip install 'dash[cloud]'\n"
        f"  cd {out}\n"
        "  # either: upload this folder at https://cloud.plotly.com\n"
        "  # or:     run the app with --debug and use the dev tools panel,\n"
        "  #         Plotly Cloud -> Sign In -> name it -> Publish App"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "cloud"))
    ap.add_argument(
        "--hub",
        default=None,
        metavar="REPO",
        help="fetch model.pt and index.npz from this HF model repo instead of "
        "bundling them, e.g. ai-vaibhavv/GeoLocator",
    )
    a = ap.parse_args()
    main(a.out, a.hub)
