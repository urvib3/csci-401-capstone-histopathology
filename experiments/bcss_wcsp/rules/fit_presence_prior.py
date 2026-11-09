# Which sets of classes appear together in a tile, counted on the fit split's ground truth.
#
# A class is "present" in a tile when it covers >= 1% of the tile's labeled pixels
# (central 256 x 256, the field the WCSP scores). With 5 classes there are 31 non-empty
# sets; each gets cost -log P(set) with add-one smoothing, so unseen sets stay possible
# but expensive. Only fit tiles are used: dev chooses settings, val is held out.
#
#   python experiments/bcss_wcsp/rules/fit_presence_prior.py

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import IGNORE, LABELS, NUM_LABELS  # noqa: E402


def present_mask(target, threshold):
    labeled = target != IGNORE
    if not labeled.any():
        return None
    share = np.bincount(target[labeled], minlength=NUM_LABELS) / labeled.sum()
    return int(sum(1 << k for k in range(NUM_LABELS) if share[k] >= threshold))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--min-labeled", type=int, default=1000)
    parser.add_argument("--out", default="experiments/bcss_wcsp/configs/tables/bcss_presence_prior.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = pipeline.dataset_for(cfg, "fit")

    counts = np.zeros(2 ** NUM_LABELS)
    for i in range(len(data)):
        target = data[i].grid_target
        if (target != IGNORE).sum() < args.min_labeled:
            continue
        mask = present_mask(target, args.threshold)
        if mask:
            counts[mask] += 1
    smoothed = counts[1:] + 1
    prob = smoothed / smoothed.sum()
    out = {
        "labels": list(LABELS),
        "threshold": args.threshold,
        "tiles": int(counts.sum()),
        "counts": {str(m): int(counts[m]) for m in range(1, 2 ** NUM_LABELS)},
        "cost": {str(m): float(-np.log(prob[m - 1])) for m in range(1, 2 ** NUM_LABELS)},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    top = sorted(range(1, 2 ** NUM_LABELS), key=lambda m: -counts[m])[:10]
    for m in top:
        names = "+".join(LABELS[k] for k in range(NUM_LABELS) if m >> k & 1)
        print(f"{int(counts[m]):6d}  {out['cost'][str(m)]:.2f}  {names}")
    print(f"{out['tiles']} fit tiles -> {args.out}")


if __name__ == "__main__":
    main()
