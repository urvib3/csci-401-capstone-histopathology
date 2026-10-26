"""WCSP refinement for a prepared PanNuke HoVer-Net patch.

Reads only predicted labels and active mask.
Ground truth is never loaded here; evaluation is handled separately.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


# ============================================================
# Repository imports
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

sys.path.insert(0, str(ROOT))

from wcsp.model.neighborhood import (  # noqa: E402
    distance_weights,
    forward_offsets,
    pair_slices,
)
from wcsp.model.problem import GridProblem  # noqa: E402
from wcsp.model.tables import HANDWRITTEN  # noqa: E402
from wcsp.solve.blocks import (  # noqa: E402
    BlockCoordinateDescent,
    BlockLayout,
)
from wcsp.solve.energy import (  # noqa: E402
    energy,
    local_costs,
)
from wcsp.solve.icm import ICMSolver  # noqa: E402


# ============================================================
# Configuration
# ============================================================

K = 5
ETA = 1

# 164 = 60 + 44 + 60
# The 44-pixel cross is wider than the maximum 2-pixel
# interaction reach.
BLOCK = 60

MAX_ROUNDS = 20

OUTPUTS = {
    "strong_fidelity": "wcsp_strong_labels.npy",
    "weaker_fidelity": "wcsp_weak_labels.npy",
}


# ============================================================
# WCSP construction
# ============================================================

def make_problem(initial, active, lam, pairwise):
    """
    Unary:
        0 if label stays equal to HoVer-Net prediction
        lambda if label changes

    Pairwise:
        HANDWRITTEN[a, b] / distance
    """

    offsets = forward_offsets()

    unary = lam * (
        np.arange(K)
        != initial[..., None]
    ).astype(float)

    weights = distance_weights(
        offsets,
        ETA,
    )

    if not pairwise:
        weights = weights * 0.0

    return GridProblem(
        unary=unary,
        offsets=offsets,
        weights=weights,
        table=HANDWRITTEN,
        active=active,
    )


def split_energy(problem, labels):
    """Return unary and pairwise portions of total energy."""

    active = problem.active_mask

    chosen = np.take_along_axis(
        problem.unary,
        labels[..., None],
        -1,
    )[..., 0]

    unary = float(
        chosen[active].sum()
    )

    pairwise = (
        energy(problem, labels)
        - unary
    )

    return unary, pairwise


# ============================================================
# Graph diagnostics
# ============================================================

def edge_counts(active, offsets):

    counts = {}

    for offset in offsets:

        p, q = pair_slices(
            active.shape,
            offset,
        )

        d2 = int(
            offset[0] ** 2
            + offset[1] ** 2
        )

        count = int(
            (
                active[p]
                & active[q]
            ).sum()
        )

        counts[d2] = (
            counts.get(d2, 0)
            + count
        )

    return dict(
        sorted(counts.items())
    )


# ============================================================
# Strong-fidelity bound
# ============================================================

def fidelity_bound(active, offsets):
    """
    Upper bound on possible pairwise savings from changing
    one pixel.

    lambda greater than this bound guarantees that changing
    a label cannot compensate for its unary penalty.
    """

    weights = distance_weights(
        offsets,
        ETA,
    )

    reach = np.zeros(
        active.shape
    )

    for offset, weight in zip(
        offsets,
        weights,
    ):

        p, q = pair_slices(
            active.shape,
            offset,
        )

        both = (
            active[p]
            & active[q]
        )

        reach[p] += (
            weight * both
        )

        reach[q] += (
            weight * both
        )

    full = (
        2
        * weights.sum()
    )

    general_bound = float(
        HANDWRITTEN.max()
        * full
    )

    data_bound = float(
        HANDWRITTEN.max()
        * reach[active].max()
    )

    return (
        general_bound,
        data_bound,
    )


# ============================================================
# Local-optimum check
# ============================================================

def is_local_minimum(problem, labels):

    costs = local_costs(
        problem,
        labels,
    )

    current = np.take_along_axis(
        costs,
        labels[..., None],
        -1,
    )[..., 0]

    return bool(
        (
            costs.min(axis=-1)
            >= current - 1e-9
        )[
            problem.active_mask
        ].all()
    )


# ============================================================
# Transition counting
# ============================================================

def transitions(before, after, active):
    """
    Count label changes such as:
        2->3
        5->3
    """

    moved = (
        active
        & (before != after)
    )

    if not moved.any():
        return {}

    before_changed = before[moved]
    after_changed = after[moved]

    pairs = np.stack(
        [
            before_changed,
            after_changed,
        ],
        axis=1,
    )

    unique_pairs, counts = np.unique(
        pairs,
        axis=0,
        return_counts=True,
    )

    return {
        f"{a}->{b}": int(n)
        for (a, b), n
        in zip(
            unique_pairs,
            counts,
        )
    }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Prepared PanNuke patch directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for WCSP outputs",
    )

    parser.add_argument(
        "--weak-lambda",
        type=float,
        default=1.0,
        help=(
            "Unary fidelity penalty used in the "
            "weak-fidelity WCSP run"
        ),
    )

    # IMPORTANT:
    # parse_args comes AFTER every add_argument.
    args = parser.parse_args()

    SOURCE = args.source.resolve()
    OUTPUT = args.output.resolve()

    weak_lambda = args.weak_lambda

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load prepared patch
    # --------------------------------------------------------

    labels = np.load(
        SOURCE
        / "hovernet_labels_164.npy"
    )

    active = np.load(
        SOURCE
        / "active_mask_164.npy"
    ).astype(bool)

    if (
        labels.shape
        != active.shape
    ):
        raise ValueError(
            "labels and active mask "
            "must have the same shape"
        )

    if not np.isin(
        labels[active],
        range(1, 6),
    ).all():
        raise ValueError(
            "active pixels must carry labels 1-5"
        )

    if (
        labels[~active]
        != 0
    ).any():
        raise ValueError(
            "inactive pixels must be 0"
        )

    # WCSP internally uses classes 0-4.
    initial = np.where(
        active,
        labels - 1,
        0,
    ).astype(np.int64)

    offsets = forward_offsets()

    edges = edge_counts(
        active,
        offsets,
    )

    general_bound, data_bound = fidelity_bound(
        active,
        offsets,
    )

    strong_lambda = (
        math.floor(
            general_bound
        )
        + 1
    )

    layout = BlockLayout.cross(
        active.shape,
        block=BLOCK,
    )

    # --------------------------------------------------------
    # Experimental settings
    # --------------------------------------------------------

    settings = [
        (
            "unary_only",
            1.0,
            False,
        ),
        (
            "strong_fidelity",
            strong_lambda,
            True,
        ),
        (
            "weaker_fidelity",
            weak_lambda,
            True,
        ),
    ]

    runs = []

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    for (
        name,
        lam,
        pairwise,
    ) in settings:

        problem = make_problem(
            initial,
            active,
            lam,
            pairwise,
        )

        solver = BlockCoordinateDescent(
            ICMSolver(),
            layout=layout,
            max_rounds=MAX_ROUNDS,
        )

        start = perf_counter()

        result = solver.solve(
            problem,
            initial,
        )

        elapsed = (
            perf_counter()
            - start
        )

        out = np.where(
            active,
            result.labels + 1,
            0,
        ).astype(
            labels.dtype
        )

        changed = int(
            (
                out != labels
            )[active].sum()
        )

        init_u, init_p = split_energy(
            problem,
            initial,
        )

        ret_u, ret_p = split_energy(
            problem,
            result.labels,
        )

        rounds = max(
            t["round"]
            for t
            in result.trace
        )

        local_min = is_local_minimum(
            problem,
            result.labels,
        )

        if name in OUTPUTS:

            np.save(
                OUTPUT
                / OUTPUTS[name],
                out,
            )

        runs.append({
            "setting":
                name,

            "lambda":
                lam,

            "pairwise_enabled":
                pairwise,

            "initial_objective":
                init_u + init_p,

            "initial_unary":
                init_u,

            "initial_pairwise":
                init_p,

            "returned_objective":
                ret_u + ret_p,

            "returned_unary":
                ret_u,

            "returned_pairwise":
                ret_p,

            "changed_pixels":
                changed,

            "changes_by_transition":
                transitions(
                    labels,
                    out,
                    active,
                ),

            "runtime_seconds":
                elapsed,

            "status": (
                "local optimum: "
                "block coordinate descent "
                "with ICM; "
                "no single-pixel improvement "
                f"left = {local_min}"
            ),

            "optimization_rounds":
                rounds,

            "energy_trace": [
                {
                    "round":
                        t["round"],
                    "phase":
                        t["phase"],
                    "energy":
                        t["energy"],
                }
                for t
                in result.trace
            ],
        })

        print(
            f"{name}: "
            f"lambda={lam}, "
            f"initial="
            f"{init_u + init_p:.4f}, "
            f"returned="
            f"{ret_u + ret_p:.4f}, "
            f"changed={changed}, "
            f"rounds={rounds}, "
            f"local_min={local_min}, "
            f"seconds={elapsed:.2f}"
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {

        "source_image":
            SOURCE.name,

        "inputs": [
            str(
                SOURCE
                / "hovernet_labels_164.npy"
            ),
            str(
                SOURCE
                / "active_mask_164.npy"
            ),
        ],

        "ground_truth_used":
            False,

        "label_dictionary": {
            "0":
                "Background / inactive",
            "1":
                "Neoplastic",
            "2":
                "Inflammatory",
            "3":
                "Connective",
            "4":
                "Dead",
            "5":
                "Non-neoplastic epithelial",
        },

        "domain": [
            1,
            2,
            3,
            4,
            5,
        ],

        "variable_count":
            int(active.sum()),

        "edge_count":
            sum(edges.values()),

        "edges_by_squared_distance": {
            str(k): v
            for k, v
            in edges.items()
        },

        "unary": (
            "lambda if a pixel changes "
            "from its HoVer-Net label, "
            "else 0"
        ),

        "pairwise": (
            "HANDWRITTEN table "
            "/ distance^eta"
        ),

        "distance_exponent_eta":
            ETA,

        "weak_lambda":
            weak_lambda,

        "neighbor_types": (
            "all five "
            "(squared distances "
            "1, 2, 4, 5, 8), "
            "each pair once"
        ),

        "boundary_handling": (
            "pairs touching an inactive "
            "pixel are omitted"
        ),

        "strong_fidelity_bound": {
            "general":
                general_bound,

            "on_this_mask":
                data_bound,
        },

        "solver": {
            "method":
                "BlockCoordinateDescent(ICMSolver)",

            "layout": (
                f"four {BLOCK}x{BLOCK} corners + "
                f"{active.shape[0] - 2 * BLOCK} "
                "px cross"
            ),

            "max_rounds":
                MAX_ROUNDS,
        },

        "runs":
            runs,
    }

    (
        OUTPUT
        / "results.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    print(
        "\nWCSP runs complete."
    )


if __name__ == "__main__":
    main()