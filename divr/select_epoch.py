#!/usr/bin/env python3
"""The epoch of a verifier chosen by its held-out within-question pair accuracy (train_verifier --holdout writes
<out_dir>/verifiers/<name>/holdout.json): the highest accuracy, ties to the earlier epoch. Prints the epoch index.

    --epoch "$(python -m divr.select_epoch --name divr_gsm8k --out_dir outputs)"
"""
import argparse
import json
import pathlib
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--out_dir", default="outputs")
args = ap.parse_args()
acc = json.load(open(pathlib.Path(args.out_dir) / "verifiers" / args.name / "holdout.json"))
best = max(sorted(acc, key=int), key=lambda e: acc[e])
print(best)
print(f"held-out pair accuracy by epoch: {acc}", file=sys.stderr)
