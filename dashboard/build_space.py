"""Assemble a self-contained Hugging Face Space directory from this repo.

The Space is a separate git repo, so it needs its own copy of everything the
app touches: the entrypoint, the model code under src/, the checkpoint, and the
precomputed cells. This script gathers them into dashboard/space/ so that
directory can be pushed as-is.

    python dashboard/prepare.py      # once, to write assets/cells.npz
    python dashboard/build_space.py
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Space-root name -> source path. src/ is copied whole; it is 66 KB and copying
# it entire means the imports in inference.py need no rewriting.
FILES = {
    "app.py": os.path.join(HERE, "app.py"),
    "inference.py": os.path.join(HERE, "inference.py"),
    "theme.py": os.path.join(HERE, "theme.py"),
    "Dockerfile": os.path.join(HERE, "Dockerfile"),
    "requirements.txt": os.path.join(HERE, "requirements-space.txt"),
    "README.md": os.path.join(HERE, "SPACE_README.md"),
    ".gitattributes": os.path.join(HERE, "space.gitattributes"),
    "model.pt": os.path.join(ROOT, "model.pt"),
    "model_assets/cells.npz": os.path.join(HERE, "model_assets", "cells.npz"),
    "assets/favicon.ico": os.path.join(HERE, "assets", "favicon.ico"),
    "assets/favicon.png": os.path.join(HERE, "assets", "favicon.png"),
    "assets/apple-touch-icon.png": os.path.join(
        HERE, "assets", "apple-touch-icon.png"
    ),
}

# Copied when it exists. Without it the Space simply does not offer the
# "retrieval" rule; see prepare.py --images.
OPTIONAL = {
    "model_assets/index.npz": os.path.join(HERE, "model_assets", "index.npz"),
    "model_assets/protocol_metrics.json": os.path.join(
        HERE, "model_assets", "protocol_metrics.json"
    ),
}


def main(out):
    """Copy the Space's files into `out`, replacing what is already there.

    Args:
        out: the directory to assemble into.

    Returns:
        None; prints a manifest and the push instructions.
    """
    missing = [s for s in FILES.values() if not os.path.exists(s)]
    if missing:
        for m in missing:
            print(f"missing: {m}", file=sys.stderr)
        raise SystemExit(
            "run `python dashboard/prepare.py` first, and check model.pt is present"
        )

    os.makedirs(out, exist_ok=True)
    for name, src in {**FILES, **OPTIONAL}.items():
        if not os.path.exists(src):
            print(f"skipping {name} (not built); see dashboard/README.md")
            continue
        dst = os.path.join(out, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    src_dst = os.path.join(out, "src")
    shutil.rmtree(src_dst, ignore_errors=True)
    shutil.copytree(
        os.path.join(ROOT, "src"), src_dst, ignore=shutil.ignore_patterns("__pycache__")
    )

    total = 0
    print(f"\n{out}")
    for dirpath, dirnames, filenames in os.walk(out):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in sorted(filenames):
            p = os.path.join(dirpath, f)
            size = os.path.getsize(p)
            total += size
            print(f"  {os.path.relpath(p, out):<28} {size / 1024:>9,.1f} KB")
    print(f"  {'total':<28} {total / 1024 / 1024:>9,.1f} MB")

    print(
        f"\nto deploy (create the Space first, SDK=Docker, hardware=CPU basic):\n"
        f"  cd {out}\n"
        f"  git init && git lfs install\n"
        f"  git lfs track '*.pt' '*.npz'\n"
        f"  git add -A && git commit -m 'geolocator dashboard'\n"
        f"  git remote add origin https://huggingface.co/spaces/<user>/<space>\n"
        f"  git push -u origin main"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "space"))
    main(ap.parse_args().out)
