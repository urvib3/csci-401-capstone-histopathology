# Upper bound for the presence rule on the 185 BCSS dev tiles (half-resolution U-Net,
# same setup as runs r_control / r_g3000): WCSP restricted to the TRUE class set of each
# tile (classes covering >= 1% of its labeled pixels) instead of the set presence picks.
# The gap between this and WCSP + presence is the cost of picking the wrong class set.
#
#   python experiments/bcss_wcsp/rules/presence_oracle_dev.py --workers 16

import argparse
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.data import select  # noqa: E402
from wcsp.evaluate.pixel import pixel_confusion, pixel_report  # noqa: E402
from wcsp.labels import IGNORE, NUM_LABELS  # noqa: E402
from wcsp.solve.factory import get_solver  # noqa: E402

CONFIG = "experiments/bcss_wcsp/configs/bcss_rules.yaml"
WCSP = {"name": "blocks", "block": 100, "max_rounds": 4, "inner": {"name": "icm"}}


def solve(i):
    cfg = load_config(CONFIG)
    data, store = pipeline.dataset_for(cfg, "dev"), pipeline.store_for(cfg, "dev")
    sample = data[i]
    if sample.id not in store or not (sample.grid_target != IGNORE).any():
        return None
    target = sample.grid_target
    problem = pipeline.problem_for(cfg, sample, store.load(sample.id), 1.0)
    ok = target != IGNORE
    allowed = np.bincount(target[ok], minlength=NUM_LABELS)[:NUM_LABELS] / ok.sum() >= 0.01
    sub = replace(problem, unary=problem.unary + 1e3 * ~allowed)
    labels = get_solver(dict(WCSP)).solve(sub, sub.unary.argmin(-1)).labels
    return pixel_confusion(labels, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    cfg = load_config(CONFIG)
    n = len(pipeline.dataset_for(cfg, "dev"))
    with Pool(args.workers) as pool:
        rows = [r for r in pool.map(solve, select(n, limit=200), chunksize=1) if r is not None]
    rep = pixel_report(sum(rows))
    out = {"tiles": len(rows), "mIoU": round(rep["mIoU"], 4)}
    Path("/scratch1/ubhuwani/wcsp/runs/presence_oracle_dev").mkdir(parents=True, exist_ok=True)
    Path("/scratch1/ubhuwani/wcsp/runs/presence_oracle_dev/report.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
