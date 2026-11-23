# Upper bounds for the two training losses, on Ryan's 185 frozen full-res dev inputs
# (probabilities + ground truth), before any training:
#
#   region_oracle    average the U-Net probabilities inside each ground-truth connected
#                    region, then argmax: what perfectly coherent regions would score
#                    (best case for the region-consistency loss, arm B)
#   presence_oracle  give the presence step the TRUE class set of the tile (classes >= 1% of
#                    labeled pixels), then the usual WCSP: what perfect class presence would
#                    score (best case for the presence loss, arm C)
#
#   python experiments/bcss_wcsp/rules/oracle_tests.py --workers 8

import argparse
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp.config import load_config  # noqa: E402
from wcsp.evaluate.pixel import pixel_confusion, pixel_report  # noqa: E402
from wcsp.labels import IGNORE, NUM_LABELS  # noqa: E402
from wcsp.model.problem import build_problem  # noqa: E402
from wcsp.solve.factory import get_solver  # noqa: E402

INPUTS = Path("data/fullres_multiscale_20261115/inputs")
WCSP = {"name": "blocks", "block": 100, "max_rounds": 4, "inner": {"name": "icm"}}


def region_oracle(probs, target):
    out = probs.argmax(0)
    for c in range(NUM_LABELS):
        comp, n = ndimage.label(target == c, structure=np.ones((3, 3)))
        for r in range(1, n + 1):
            region = comp == r
            out[region] = probs[:, region].mean(1).argmax()
    return out


def solve_tile(path):
    z = np.load(path)
    probs, target = z["probs"].astype(np.float64), z["target"]
    raw = probs.argmax(0)
    out = {"raw": pixel_confusion(raw, target), "region_oracle": pixel_confusion(region_oracle(probs, target), target)}
    ok = target != IGNORE
    share = np.bincount(target[ok], minlength=NUM_LABELS)[:NUM_LABELS] / max(ok.sum(), 1)
    allowed = share >= 0.01
    problem = build_problem(probs, load_config("experiments/bcss_wcsp/configs/bcss_rules.yaml").problem)
    sub = replace(problem, unary=problem.unary + 1e3 * ~allowed)
    labels = get_solver(dict(WCSP)).solve(sub, sub.unary.argmin(-1)).labels
    out["presence_oracle"] = pixel_confusion(labels, target)
    out["truth_classes"] = int(allowed.sum())
    out["raw_classes"] = int((np.bincount(raw[ok], minlength=NUM_LABELS)[:NUM_LABELS] / max(ok.sum(), 1) >= 0.01).sum())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="/scratch1/ubhuwani/wcsp/runs/fullres_oracles_dev")
    args = parser.parse_args()
    with Pool(args.workers) as pool:
        rows = pool.map(solve_tile, sorted(INPUTS.glob("*.npz")), chunksize=1)
    report = {"tiles": len(rows),
              "mean_classes_truth": float(np.mean([r["truth_classes"] for r in rows])),
              "mean_classes_raw": float(np.mean([r["raw_classes"] for r in rows]))}
    for arm in ("raw", "region_oracle", "presence_oracle"):
        rep = pixel_report(sum(r[arm] for r in rows))
        report[arm] = {k: round(v, 4) for k, v in rep.items() if k == "mIoU" or k.startswith("IoU_")}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
