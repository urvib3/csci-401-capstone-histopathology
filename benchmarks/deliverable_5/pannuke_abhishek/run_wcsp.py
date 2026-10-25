"""WCSP refinement of the full HoVer-Net prediction for PanNuke image_00748.

Reads only Aakanksha's predicted labels and active mask; the ground truth is
never loaded here (see evaluate.py).
"""

import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE.parent / "pannuke_aakanksha"
sys.path.insert(0, str(ROOT))

from wcsp.model.neighborhood import distance_weights, forward_offsets, pair_slices  # noqa: E402
from wcsp.model.problem import GridProblem  # noqa: E402
from wcsp.model.tables import HANDWRITTEN  # noqa: E402
from wcsp.solve.blocks import BlockCoordinateDescent, BlockLayout  # noqa: E402
from wcsp.solve.energy import energy, local_costs  # noqa: E402
from wcsp.solve.icm import ICMSolver  # noqa: E402

K = 5
ETA = 1
BLOCK = 60          # 164 = 60 + 44 + 60; the 44 px cross exceeds the 2 px reach
MAX_ROUNDS = 20
WEAK_LAMBDA = 1     # same as D4's weaker-fidelity run
OUTPUTS = {"strong_fidelity": "wcsp_strong_labels.npy",
           "weaker_fidelity": "wcsp_weak_labels.npy"}


def make_problem(initial, active, lam, pairwise):
    """lam per changed pixel + HANDWRITTEN[a, b] / d over active pairs."""
    offsets = forward_offsets()
    unary = lam * (np.arange(K) != initial[..., None]).astype(float)
    weights = distance_weights(offsets, ETA) * (1.0 if pairwise else 0.0)
    return GridProblem(unary=unary, offsets=offsets, weights=weights,
                       table=HANDWRITTEN, active=active)


def split_energy(problem, labels):
    """(unary, pairwise) parts of the energy over active pixels."""
    active = problem.active_mask
    chosen = np.take_along_axis(problem.unary, labels[..., None], -1)[..., 0]
    unary = float(chosen[active].sum())
    return unary, energy(problem, labels) - unary


def edge_counts(active, offsets):
    counts = {}
    for offset in offsets:
        p, q = pair_slices(active.shape, offset)
        d2 = int(offset[0] ** 2 + offset[1] ** 2)
        counts[d2] = counts.get(d2, 0) + int((active[p] & active[q]).sum())
    return dict(sorted(counts.items()))


def fidelity_bound(active, offsets):
    """
    Largest pairwise saving from changing one pixel: max table entry times the
    largest sum of 1/d over a pixel's active neighbors. Changing any set S of
    pixels saves at most |S| times this, so lambda above it keeps HoVer-Net.
    """
    weights = distance_weights(offsets, ETA)
    reach = np.zeros(active.shape)
    for offset, w in zip(offsets, weights):
        p, q = pair_slices(active.shape, offset)
        both = active[p] & active[q]
        reach[p] += w * both
        reach[q] += w * both
    full = 2 * weights.sum()
    return float(HANDWRITTEN.max() * full), float(HANDWRITTEN.max() * reach[active].max())


def is_local_minimum(problem, labels):
    """No single active pixel can lower the energy by changing its label."""
    costs = local_costs(problem, labels)
    current = np.take_along_axis(costs, labels[..., None], -1)[..., 0]
    return bool((costs.min(axis=-1) >= current - 1e-9)[problem.active_mask].all())


def main():
    labels = np.load(SOURCE / "hovernet_labels_164.npy")
    active = np.load(SOURCE / "active_mask_164.npy").astype(bool)
    if labels.shape != active.shape or not np.isin(labels[active], range(1, 6)).all():
        raise ValueError("active pixels must carry labels 1-5")
    if (labels[~active] != 0).any():
        raise ValueError("inactive pixels must be 0")

    initial = np.where(active, labels - 1, 0).astype(np.int64)   # wcsp labels 0-4
    offsets = forward_offsets()
    edges = edge_counts(active, offsets)
    general_bound, data_bound = fidelity_bound(active, offsets)
    strong_lambda = math.floor(general_bound) + 1
    layout = BlockLayout.cross(active.shape, block=BLOCK)

    settings = [("unary_only", 1, False),
                ("strong_fidelity", strong_lambda, True),
                ("weaker_fidelity", WEAK_LAMBDA, True)]
    runs = []
    for name, lam, pairwise in settings:
        problem = make_problem(initial, active, lam, pairwise)
        solver = BlockCoordinateDescent(ICMSolver(), layout=layout, max_rounds=MAX_ROUNDS)
        start = perf_counter()
        result = solver.solve(problem, initial)
        elapsed = perf_counter() - start

        out = np.where(active, result.labels + 1, 0).astype(labels.dtype)
        changed = int((out != labels)[active].sum())
        init_u, init_p = split_energy(problem, initial)
        ret_u, ret_p = split_energy(problem, result.labels)
        rounds = max(t["round"] for t in result.trace)
        local_min = is_local_minimum(problem, result.labels)
        if name in OUTPUTS:
            np.save(HERE / OUTPUTS[name], out)
        runs.append({
            "setting": name, "lambda": lam, "pairwise_enabled": pairwise,
            "initial_objective": init_u + init_p,
            "initial_unary": init_u, "initial_pairwise": init_p,
            "returned_objective": ret_u + ret_p,
            "returned_unary": ret_u, "returned_pairwise": ret_p,
            "changed_pixels": changed,
            "changes_by_transition": transitions(labels, out, active),
            "runtime_seconds": elapsed,
            "status": ("local optimum: block coordinate descent with ICM; "
                       f"no single-pixel improvement left = {local_min}"),
            "optimization_rounds": rounds,
            "energy_trace": [{"round": t["round"], "phase": t["phase"], "energy": t["energy"]}
                             for t in result.trace],
        })
        print(f"{name}: lambda={lam}, initial={init_u + init_p:.4f}, "
              f"returned={ret_u + ret_p:.4f}, changed={changed}, rounds={rounds}, "
              f"local_min={local_min}, seconds={elapsed:.2f}")

    report = {
        "source_image": "image_00748",
        "inputs": ["pannuke_aakanksha/hovernet_labels_164.npy",
                   "pannuke_aakanksha/active_mask_164.npy"],
        "ground_truth_used": False,
        "label_dictionary": {"0": "Background / inactive", "1": "Neoplastic",
                             "2": "Inflammatory", "3": "Connective", "4": "Dead",
                             "5": "Non-neoplastic epithelial"},
        "domain": [1, 2, 3, 4, 5],
        "variable_count": int(active.sum()),
        "edge_count": sum(edges.values()),
        "edges_by_squared_distance": {str(k): v for k, v in edges.items()},
        "unary": "lambda if a pixel changes from its HoVer-Net label, else 0",
        "pairwise": "HANDWRITTEN table (wcsp/model/tables.py, same as D4) / distance^eta",
        "distance_exponent_eta": ETA,
        "neighbor_types": "all five (squared distances 1, 2, 4, 5, 8), each pair once",
        "boundary_handling": "pairs touching an inactive pixel are omitted, as in D4",
        "strong_fidelity_bound": {
            "general": general_bound,
            "on_this_mask": data_bound,
            "argument": ("changing one pixel lowers the pairwise cost by at most "
                         "max(table) * sum of 1/d over its <= 24 neighbors; changing a set S "
                         "lowers it by at most |S| times that, while costing lambda * |S|"),
        },
        "solver": {"method": "BlockCoordinateDescent(ICMSolver)",
                   "layout": f"four {BLOCK}x{BLOCK} corners + "
                             f"{active.shape[0] - 2 * BLOCK} px cross",
                   "max_rounds": MAX_ROUNDS},
        "runs": runs,
    }
    (HERE / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def transitions(before, after, active):
    """Counts of 'a->b' over active pixels whose label changed."""
    moved = active & (before != after)
    pairs, counts = np.unique(np.stack([before[moved], after[moved]]), axis=1,
                              return_counts=True)
    return {f"{a}->{b}": int(n) for (a, b), n in zip(pairs.T, counts)}


if __name__ == "__main__":
    main()
