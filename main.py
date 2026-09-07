"""Train, then write predictions.csv for the unlabelled test images.

python main.py --train splits/train.csv --val splits/val.csv
"""

import argparse
import os

from src import config as cfg
from src import predict
from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True, help="csv of labelled training rows")
parser.add_argument("--val", required=True, help="csv to early-stop on")
parser.add_argument("--test", default=None, help="csv scored once at the end")
parser.add_argument(
    "--img-dir",
    default=cfg.IMG_DIR,
    help="directory holding the labelled training images",
)
parser.add_argument(
    "--pretrained",
    default=cfg.PRETRAINED_PATH,
    help="imagenet weights for the backbone, or 'none' to train scratch",
)
parser.add_argument(
    "--images",
    default="geo_dataset/holdout_public",
    help="directory of unlabelled images to predict",
)
parser.add_argument("--out", default=cfg.OUT_DIR, help="output directory")
parser.add_argument("--predictions", default="predictions.csv")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

cfg.IMG_DIR = args.img_dir
cfg.PRETRAINED_PATH = (
    None
    if not args.pretrained or args.pretrained.lower() == "none"
    else args.pretrained
)
if args.seed is not None:
    cfg.SEED = args.seed

for d in (args.img_dir, args.images):
    if not os.path.isdir(d):
        raise SystemExit(f"no such image directory: {d}")

train(cfg, args.train, args.val, args.test, args.out)
predict.run(
    cfg, f"{args.out}/model.pt", args.train, args.predictions, images=args.images
)
