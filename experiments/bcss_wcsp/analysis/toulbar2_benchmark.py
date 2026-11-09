# How long toulbar2 takes, and how far its search gets, on the WCSP's real blocks.
#
# For a few dev tiles (data-derived b, lambda = 8), condition each region of the
# 100/56/100 block scheme on the U-Net labels (the first step of the real solver) and
# solve it with ICM and with toulbar2 at a given time limit. Records per block:
# seconds, search nodes, backtracks, whether toulbar2 proved optimality (dual bound
# reaches its energy), and its energy relative to ICM's.
#
#   python experiments/bcss_wcsp/analysis/toulbar2_benchmark.py --time-limit 20 --tiles 8

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.data import select  # noqa: E402
from wcsp.solve.argmax import ArgmaxSolver  # noqa: E402
from wcsp.solve.blocks import BlockLayout  # noqa: E402
from wcsp.solve.condition import condition  # noqa: E402
from wcsp.solve.energy import energy  # noqa: E402
from wcsp.solve.icm import ICMSolver  # noqa: E402
from wcsp.solve.toulbar2 import Toulbar2Solver  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_presence.yaml")
    parser.add_argument("--time-limit", type=float, default=20)
    parser.add_argument("--tiles", type=int, default=8)
    parser.add_argument("--out", default="experiments/bcss_wcsp/analysis/toulbar2_benchmark")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = pipeline.dataset_for(cfg, "dev")
    store = pipeline.store_for(cfg, "dev")
    positions = [i for i in select(len(data), limit=200) if data.ids[i] in store]
    step = max(1, len(positions) // args.tiles)
    toulbar = Toulbar2Solver(time_limit=args.time_limit)

    rows = []
    for i in positions[::step][:args.tiles]:
        sample = data[i]
        problem = pipeline.problem_for(cfg, sample, store.load(sample.id), 1.0)
        labels = ArgmaxSolver().solve(problem, np.zeros(problem.shape, dtype=np.int64)).labels
        layout = BlockLayout.cross(problem.shape)
        for phase, regions in enumerate(layout.phases):
            for b, region in enumerate(regions):
                sub, window = condition(problem, labels, region)
                start_labels = labels[window].copy()
                e0 = energy(sub, start_labels)
                icm = ICMSolver().solve(sub, start_labels)
                cfn, flat = toulbar.build(sub, start_labels)
                t = time.perf_counter()
                found = cfn.Solve(showSolutions=0, timeLimit=int(np.ceil(args.time_limit)))
                seconds = time.perf_counter() - t
                e_tb = e0
                if found:
                    cand = start_labels.copy()
                    cand.ravel()[flat] = np.asarray(found[0], dtype=np.int64)
                    e_tb = min(e0, energy(sub, cand))
                try:
                    bound = sub.constant + float(cfn.GetDDualBound())
                except Exception:
                    bound = float("nan")
                rows.append({
                    "id": sample.id, "block": "corner" if phase == 0 else "cross",
                    "variables": int(sub.num_variables), "seconds": seconds,
                    "nodes": int(cfn.GetNbNodes()), "backtracks": int(cfn.GetNbBacktracks()),
                    "found": bool(found), "proved_optimal": bool(found) and bound >= e_tb - 1e-3 * max(1, abs(e_tb)),
                    "energy_start": e0, "energy_icm": icm.energy, "energy_toulbar2": e_tb,
                    "icm_seconds": icm.seconds,
                })
                print(rows[-1], flush=True)

    out = Path(f"{args.out}_{args.time_limit:g}s.json")
    out.write_text(json.dumps({"time_limit": args.time_limit, "blocks": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
