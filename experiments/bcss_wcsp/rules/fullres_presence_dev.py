# Class presence on the full-resolution U-Net: Ryan's 185 frozen dev inputs
# (data/fullres_multiscale_20261115/inputs, 512 x 512 probabilities + targets, Git LFS).
#
# Arms, all scored on the same tiles:
#   raw        U-Net argmax
#   wcsp       pixel WCSP (data-derived b, lambda = 8, 100 px ICM blocks), as Ryan's mu = 0 control
#   presence   class presence, then the same WCSP
#   clusters_presence   class presence, then cluster WCSP (SLIC on the U-Net probabilities,
#              1,024 clusters = the 256-tile setting's cluster size at 4x the area) + pixel refine
#
# gamma is the half-resolution choice (3000 on 256 x 256) scaled by tile area (x 4 = 12000),
# so the per-pixel strength is unchanged and nothing is re-tuned. The class-set prior is the
# existing fit prior (same 128 um field of view); --prior can point to a v2 refit.
#
#   python experiments/bcss_wcsp/rules/fullres_presence_dev.py --workers 16

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp.config import load_config  # noqa: E402
from wcsp.evaluate.pixel import pixel_confusion, pixel_report  # noqa: E402
from wcsp.model.problem import build_problem  # noqa: E402
from wcsp.solve.factory import get_solver  # noqa: E402

INPUTS = Path("data/fullres_multiscale_20261115/inputs")
CONFIG = "experiments/bcss_wcsp/configs/bcss_rules.yaml"
PRIOR = "experiments/bcss_wcsp/configs/tables/bcss_presence_prior.json"
WCSP = {"name": "blocks", "block": 100, "max_rounds": 4, "inner": {"name": "icm"}}
ARGS = None
ARMS = ("raw", "wcsp", "presence", "clusters_presence")


def solve_tile(path):
    z = np.load(path)
    probs, target = z["probs"].astype(np.float64), z["target"]
    cfg = load_config(CONFIG)
    problem = build_problem(probs, cfg.problem)
    init = problem.unary.argmin(-1)
    out = {"id": str(z["id"]), "raw": pixel_confusion(init, target)}
    out["wcsp"] = pixel_confusion(get_solver(dict(WCSP)).solve(problem, init).labels, target)
    presence = get_solver({"name": "presence", "prior": ARGS.prior, "gamma": ARGS.gamma, "inner": dict(WCSP)})
    result = presence.solve(problem, init)
    out["presence"] = pixel_confusion(result.labels, target)
    out["dropped"] = int(result.info["dropped"])
    if ARGS.clusters:
        clusters = {"name": "clusters", "n_segments": 1024, "compactness": 10.0, "features": "probs",
                    "prob_weight": 30.0, "refine": dict(WCSP)}
        both = get_solver({"name": "presence", "prior": ARGS.prior, "gamma": ARGS.gamma, "inner": clusters})
        out["clusters_presence"] = pixel_confusion(both.solve(problem, init).labels, target)
    return out


def init_worker(args):
    global ARGS
    ARGS = args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=12000.0)
    parser.add_argument("--prior", default=PRIOR)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clusters", action="store_true", help="also run cluster WCSP + presence")
    parser.add_argument("--out", default="/scratch1/ubhuwani/wcsp/runs/fullres_presence_dev")
    args = parser.parse_args()
    paths = sorted(INPUTS.glob("*.npz"))[: args.limit or None]
    with Pool(args.workers, initializer=init_worker, initargs=(args,)) as pool:
        rows = pool.map(solve_tile, paths, chunksize=1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "confusions.npz", ids=np.array([r["id"] for r in rows]),
                        **{arm: np.stack([r[arm] for r in rows]) for arm in ARMS if arm in rows[0]})
    report = {"tiles": len(rows), "gamma": args.gamma, "prior": args.prior,
              "tiles_with_class_dropped": int(sum(r["dropped"] > 0 for r in rows))}
    for arm in (a for a in ARMS if a in rows[0]):
        rep = pixel_report(sum(r[arm] for r in rows))
        report[arm] = {k: round(v, 4) for k, v in rep.items() if k == "mIoU" or k.startswith("IoU_")}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
