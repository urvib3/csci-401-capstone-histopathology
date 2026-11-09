# What kind of mistakes does the U-Net make, and which could a smoothing WCSP fix?
#
# For every labeled val pixel the U-Net gets wrong, record
#   - how far it is from the nearest U-Net label boundary, and
#   - the size of the connected U-Net region it sits in.
# Errors deep inside large predicted regions are "region errors": no local
# rule can see them, since every neighbor agrees with the wrong label.
#
# Also an oracle ceiling for island removal: relabel every predicted region
# smaller than A pixels with its majority ground-truth label (or leave it if
# unlabeled). No smoother that only removes small islands can beat this.
#
#   python experiments/bcss_wcsp/analysis/error_structure.py --run valfull_hw_lam8 --split val

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.evaluate.pixel import pixel_confusion, pixel_report  # noqa: E402
from wcsp.labels import IGNORE, NUM_LABELS  # noqa: E402

DIST_BINS = [0, 2, 5, 10, 20, np.inf]          # px from a predicted boundary
SIZE_BINS = [0, 64, 1024, 8192, np.inf]        # px in the predicted region
ISLANDS = [64, 256, 1024]


def regions(labels):
    """(H, W) size of the 8-connected same-label region each pixel belongs to, and region ids."""
    sizes = np.zeros(labels.shape, dtype=np.int64)
    ids = np.zeros(labels.shape, dtype=np.int64)
    offset = 0
    for k in range(NUM_LABELS):
        comp, n = ndimage.label(labels == k, structure=np.ones((3, 3)))
        if n:
            counts = np.bincount(comp.ravel())
            mask = comp > 0
            sizes[mask] = counts[comp[mask]]
            ids[mask] = comp[mask] + offset
            offset += n
    return sizes, ids


def boundary_distance(labels):
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[1:] |= labels[:-1] != labels[1:]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    if not edge.any():
        return np.full(labels.shape, np.inf)
    return ndimage.distance_transform_edt(~edge)


def island_oracle(pred, target, ids, sizes, limit):
    out = pred.copy()
    small = (sizes < limit) & (sizes > 0)
    for rid in np.unique(ids[small]):
        mask = ids == rid
        truth = target[mask]
        truth = truth[truth != IGNORE]
        if len(truth):
            out[mask] = np.bincount(truth, minlength=NUM_LABELS).argmax()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--run", default="valfull_hw_lam8")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="experiments/bcss_wcsp/analysis/error_structure_val.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = pipeline.dataset_for(cfg, args.split)
    run = cfg.runs_root / args.run / args.split

    dist_hist = {m: np.zeros((2, len(DIST_BINS) - 1)) for m in ("unet",)}
    size_hist = np.zeros((2, len(SIZE_BINS) - 1))          # row 0 errors, row 1 all labeled pixels
    fixed = np.zeros(len(DIST_BINS) - 1)
    broken = np.zeros(len(DIST_BINS) - 1)
    confusions = {"unet": 0, "wcsp": 0, **{f"oracle_{a}": 0 for a in ISLANDS}}
    for shard in sorted(run.glob("shard_*.npz")):
        z = np.load(shard)
        for sample_id, pred, refined in zip(z["ids"].tolist(), z["argmax"], z["wcsp"]):
            target = data[data.index(sample_id)].grid_target
            labeled = target != IGNORE
            wrong = labeled & (pred != target)
            dist = boundary_distance(pred)
            sizes, ids = regions(pred)
            d_bin = np.digitize(dist, DIST_BINS[1:-1])
            s_bin = np.digitize(sizes, SIZE_BINS[1:-1])
            dist_hist["unet"][0] += np.bincount(d_bin[wrong], minlength=len(DIST_BINS) - 1)
            dist_hist["unet"][1] += np.bincount(d_bin[labeled], minlength=len(DIST_BINS) - 1)
            size_hist[0] += np.bincount(s_bin[wrong], minlength=len(SIZE_BINS) - 1)
            size_hist[1] += np.bincount(s_bin[labeled], minlength=len(SIZE_BINS) - 1)
            now_right = labeled & (refined == target)
            fixed += np.bincount(d_bin[wrong & now_right], minlength=len(DIST_BINS) - 1)
            broken += np.bincount(d_bin[labeled & ~wrong & ~now_right], minlength=len(DIST_BINS) - 1)
            confusions["unet"] = confusions["unet"] + pixel_confusion(pred, target)
            confusions["wcsp"] = confusions["wcsp"] + pixel_confusion(refined, target)
            for a in ISLANDS:
                confusions[f"oracle_{a}"] = confusions[f"oracle_{a}"] + pixel_confusion(
                    island_oracle(pred, target, ids, sizes, a), target)

    def label(bins, i, unit):
        return f"{bins[i]:g}–{bins[i + 1]:g} {unit}" if np.isfinite(bins[i + 1]) else f"≥ {bins[i]:g} {unit}"

    errors = dist_hist["unet"][0].sum()
    report = {
        "run": args.run,
        "labeled_pixels": float(size_hist[1].sum()),
        "unet_errors": float(errors),
        "errors_by_boundary_distance": {
            label(DIST_BINS, i, "px"): {
                "share_of_errors": float(dist_hist["unet"][0][i] / errors),
                "error_rate": float(dist_hist["unet"][0][i] / max(dist_hist["unet"][1][i], 1)),
                "wcsp_fixed": float(fixed[i]), "wcsp_broken": float(broken[i]),
            } for i in range(len(DIST_BINS) - 1)},
        "errors_by_region_size": {
            label(SIZE_BINS, i, "px"): {
                "share_of_errors": float(size_hist[0][i] / errors),
                "error_rate": float(size_hist[0][i] / max(size_hist[1][i], 1)),
            } for i in range(len(SIZE_BINS) - 1)},
        "scores": {name: {k: v for k, v in pixel_report(c).items() if k in ("pixel_acc", "mIoU")}
                   for name, c in confusions.items()},
        "unet_confusion": np.asarray(confusions["unet"]).tolist(),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("unet_errors", "errors_by_boundary_distance",
                                             "errors_by_region_size", "scores")}, indent=2))


if __name__ == "__main__":
    main()
