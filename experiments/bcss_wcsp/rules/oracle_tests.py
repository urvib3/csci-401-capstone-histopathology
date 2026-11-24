# Upper bounds for the two problems, on the 185 frozen full-res BCSS dev inputs
# (U-Net probabilities + ground truth). No WCSP: each bound changes only the U-Net output.
#
#   perfect_class_id   keep the U-Net's own regions (8-connected pieces of each predicted
#                      class) and give each region the ground-truth class most of its labeled
#                      pixels have: perfect class identification, U-Net segmentation
#   perfect_seg        keep the true regions (8-connected pieces of each ground-truth class)
#                      and give each region the class the U-Net predicts for most of it:
#                      perfect segmentation, U-Net class identification
#   both               the ground truth (mIoU 1.0 by construction)
#
#   python experiments/bcss_wcsp/rules/oracle_tests.py --workers 8

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp.evaluate.pixel import pixel_confusion, pixel_report  # noqa: E402
from wcsp.labels import IGNORE, NUM_LABELS  # noqa: E402

INPUTS = Path("data/fullres_multiscale_20261115/inputs")
EIGHT = np.ones((3, 3))


def relabel_regions(regions_from, votes_from, valid):
    """Each 8-connected piece of each class in regions_from gets the majority of votes_from
    over its labeled pixels (pieces with no labeled pixel keep their class)."""
    out = regions_from.copy()
    for c in range(NUM_LABELS):
        comp, n = ndimage.label(regions_from == c, structure=EIGHT)
        for r in range(1, n + 1):
            region = comp == r
            votes = votes_from[region & valid]
            if votes.size:
                out[region] = np.bincount(votes, minlength=NUM_LABELS)[:NUM_LABELS].argmax()
    return out


def region_oracle(probs, target):
    """Perfect segmentation: true regions, labeled with the U-Net's majority class."""
    pred = probs.argmax(0)
    valid = target != IGNORE
    return relabel_regions(np.where(valid, target, pred), pred, valid)


def solve_tile(path):
    z = np.load(path)
    target, pred = z["target"], z["probs"].astype(np.float32).argmax(0)
    valid = target != IGNORE
    return {
        "raw": pixel_confusion(pred, target),
        "perfect_class_id": pixel_confusion(relabel_regions(pred, target, valid), target),
        "perfect_seg": pixel_confusion(region_oracle(z["probs"].astype(np.float32), target), target),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="/scratch1/ubhuwani/wcsp/runs/fullres_oracles_dev")
    args = parser.parse_args()
    with Pool(args.workers) as pool:
        rows = pool.map(solve_tile, sorted(INPUTS.glob("*.npz")), chunksize=1)
    report = {"tiles": len(rows)}
    for arm in ("raw", "perfect_class_id", "perfect_seg"):
        rep = pixel_report(sum(r[arm] for r in rows))
        report[arm] = {k: round(v, 4) for k, v in rep.items() if k == "mIoU" or k.startswith("IoU_")}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "report_v2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
