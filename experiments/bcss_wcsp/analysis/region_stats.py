# Region-level statistics: are there biological regularities beyond "neighbors agree"?
#
# For a labeling (ground truth or U-Net), per class:
#   - regions per tile and median region area (8-connected, labeled pixels only),
#   - share of the class's border shared with each other class (region adjacency),
#   - P(B in tile | A in tile), a class counting as present at >= 1% of labeled pixels.
# Ground truth on fit is where rules may be read from; U-Net vs. ground truth on val shows
# which regularities the classifier actually breaks.
#
#   python experiments/bcss_wcsp/analysis/region_stats.py

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import IGNORE, NUM_LABELS  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]


class Stats:
    def __init__(self):
        self.areas = [[] for _ in range(NUM_LABELS)]
        self.border = np.zeros((NUM_LABELS, NUM_LABELS))
        self.present = np.zeros(NUM_LABELS)
        self.together = np.zeros((NUM_LABELS, NUM_LABELS))
        self.tiles = 0

    def add(self, labels):
        valid = labels != IGNORE
        if valid.sum() < 1000:
            return
        self.tiles += 1
        for k in range(NUM_LABELS):
            comp, n = ndimage.label(labels == k, structure=np.ones((3, 3)))
            if n:
                self.areas[k].extend(np.bincount(comp.ravel())[1:].tolist())
        for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
            ok = (a != IGNORE) & (b != IGNORE) & (a != b)
            np.add.at(self.border, (a[ok], b[ok]), 1)
            np.add.at(self.border, (b[ok], a[ok]), 1)
        share = np.bincount(labels[valid], minlength=NUM_LABELS) / valid.sum()
        here = share >= 0.01
        self.present += here
        self.together += np.outer(here, here)

    def report(self):
        border = self.border / np.maximum(self.border.sum(1, keepdims=True), 1)
        return {
            "tiles": self.tiles,
            "regions_per_tile": {NAMES[k]: len(self.areas[k]) / max(self.tiles, 1) for k in range(NUM_LABELS)},
            "median_region_px": {NAMES[k]: float(np.median(self.areas[k])) if self.areas[k] else None
                                 for k in range(NUM_LABELS)},
            "share_regions_under_64px": {NAMES[k]: float(np.mean(np.array(self.areas[k]) < 64))
                                         if self.areas[k] else None for k in range(NUM_LABELS)},
            "border_share": {NAMES[a]: {NAMES[b]: float(border[a, b]) for b in range(NUM_LABELS) if b != a}
                             for a in range(NUM_LABELS)},
            "p_b_given_a": {NAMES[a]: {NAMES[b]: float(self.together[a, b] / max(self.present[a], 1))
                                       for b in range(NUM_LABELS) if b != a} for a in range(NUM_LABELS)},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--run", default="valfull_data_lam8")
    parser.add_argument("--fit-limit", type=int, default=3000)
    parser.add_argument("--out", default="experiments/bcss_wcsp/analysis/region_stats.json")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = {}
    fit = pipeline.dataset_for(cfg, "fit")
    step = max(1, len(fit) // args.fit_limit)
    truth_fit = Stats()
    for i in range(0, len(fit), step):
        truth_fit.add(fit[i].grid_target)
    out["truth_fit"] = truth_fit.report()

    val = pipeline.dataset_for(cfg, "val")
    truth_val, unet_val, wcsp_val = Stats(), Stats(), Stats()
    for shard in sorted((cfg.runs_root / args.run / "val").glob("shard_*.npz")):
        z = np.load(shard)
        for sample_id, pred, refined in zip(z["ids"].tolist(), z["argmax"], z["wcsp"]):
            target = val[val.index(sample_id)].grid_target
            ignored = target == IGNORE
            truth_val.add(target)
            unet_val.add(np.where(ignored, IGNORE, pred))
            wcsp_val.add(np.where(ignored, IGNORE, refined))
    out["truth_val"] = truth_val.report()
    out["unet_val"] = unet_val.report()
    out["wcsp_val"] = wcsp_val.report()
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
