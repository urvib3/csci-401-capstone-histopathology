"""Exact, unscaled WCSP runs on the eight active pixels supplied by Task 1."""

import csv
import itertools
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "pannuke_task1"
# hist.pdf order: neoplastic, inflammatory, connective, dead, non-neoplastic.
BASE = np.array([
    [0, 2, 1, 1, 9],
    [2, 0, 2, 1, 8],
    [1, 2, 0, 5, 1],
    [1, 1, 5, 0, 10],
    [9, 8, 1, 10, 0],
], dtype=float)


def main():
    grid = np.loadtxt(SOURCE / "pannuke_task1_label_crop.csv", delimiter=",", dtype=int)
    metadata = json.loads((SOURCE / "pannuke_task1_metadata.json").read_text())
    if grid.tolist() != metadata["label_grid"]:
        raise ValueError("CSV and metadata label grids disagree")
    if not np.array_equal(grid, np.load(SOURCE / "pannuke_task1_label_crop.npy")):
        raise ValueError("CSV and NPY label grids disagree")
    pixels = [tuple(map(int, p)) for p in np.argwhere(grid != 0)]
    if len(pixels) != metadata["active_pixels"] or not np.isin(grid, range(6)).all():
        raise ValueError("Invalid active count or class IDs")
    initial = np.array([grid[p] for p in pixels])
    edges = []
    for i, (r, c) in enumerate(pixels):
        for j in range(i + 1, len(pixels)):
            rr, cc = pixels[j]
            squared_distance = (r - rr) ** 2 + (c - cc) ** 2
            if squared_distance in (1, 2, 4, 5, 8):
                edges.append((i, j, squared_distance))
    bound = sum(float(BASE.max()) / math.sqrt(d2) for _, _, d2 in edges)
    strong_lambda = math.ceil(bound) + 1
    settings = [("unary_only", 1, False),
                ("strong_fidelity", strong_lambda, True),
                ("weaker_fidelity", 1, True)]
    runs = []
    for name, weight, binary_enabled in settings:
        start = perf_counter()
        # Lexicographic order gives a reproducible representative when optima tie.
        assignments = np.array(list(itertools.product(range(1, 6), repeat=len(pixels))),
                               dtype=np.int8)
        changed = (assignments != initial).sum(axis=1)
        unary = weight * changed
        binary = np.zeros(len(assignments))
        initial_binary = 0.0
        if binary_enabled:
            for i, j, d2 in edges:
                binary += BASE[assignments[:, i] - 1, assignments[:, j] - 1] / math.sqrt(d2)
                initial_binary += BASE[initial[i] - 1, initial[j] - 1] / math.sqrt(d2)
        total = unary + binary
        minimum = float(total.min())
        winners = np.flatnonzero(np.isclose(total, minimum, rtol=0, atol=1e-9))
        best = int(winners[0])
        elapsed = perf_counter() - start
        output = grid.copy()
        for pixel, label in zip(pixels, assignments[best]):
            output[pixel] = label
        result = {
            "setting": name, "lambda": weight, "binary_enabled": binary_enabled,
            "initial_objective": initial_binary,
            "returned_unary": int(unary[best]), "returned_pairwise": float(binary[best]),
            "returned_objective": float(total[best]),
            "changed_pixels": int(changed[best]), "runtime_seconds": elapsed,
            "status": "exact optimum (exhaustive enumeration)",
            "assignments_evaluated": len(assignments), "optimal_assignment_count": len(winners),
            "returned_grid": output.tolist(),
        }
        runs.append(result)
        np.savetxt(HERE / f"{name}_labels.csv", output, delimiter=",", fmt="%d")
        print(f"{name}: initial={initial_binary:.9f}, returned={total[best]:.9f}, "
              f"changed={changed[best]}, optima={len(winners)}, seconds={elapsed:.3f}")
    report = {
        "source_image": metadata["source_image"], "initial_grid": grid.tolist(),
        "label_dictionary": metadata["label_dictionary"],
        "domain": [1, 2, 3, 4, 5], "active_pixels_row_col": pixels,
        "variable_count": len(pixels), "edge_count": len(edges),
        "edges_by_squared_distance": {str(d): sum(e[2] == d for e in edges)
                                      for d in (1, 2, 4, 5, 8)},
        "attribute_term_F": 0, "distance_exponent_eta": 1,
        "cost_scaling": "none; floating-point costs", "tie_tolerance": 1e-9,
        "strong_fidelity_bound": bound,
        "tie_rule": "lexicographically first assignment in row-major active-pixel order",
        "boundary_handling": "standalone crop; omit background and outside-crop interactions",
        "runtime_scope": "per run: assignment generation, cost evaluation, and optimum selection",
        "runs": runs,
    }
    (HERE / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (HERE / "results.csv").open("w", newline="") as stream:
        columns = [key for key in runs[0] if key != "returned_grid"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)


if __name__ == "__main__":
    main()
