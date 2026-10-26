"""Independently rebuild costs/neighbors and check all Task 2 optima."""

import csv
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Separate transcription from hist.pdf; this verifier does not import Task 2.
COST = {
    1: (0, 2, 1, 1, 9), 2: (2, 0, 2, 1, 8),
    3: (1, 2, 0, 5, 1), 4: (1, 1, 5, 0, 10),
    5: (9, 8, 1, 10, 0),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    with (ROOT / "pannuke_task1/pannuke_task1_label_crop.csv").open() as stream:
        grid = [list(map(int, row)) for row in csv.reader(stream)]
    metadata = json.loads((ROOT / "pannuke_task1/pannuke_task1_metadata.json").read_text())
    result = json.loads((ROOT / "pannuke_task2/results.json").read_text())
    require(grid == metadata["label_grid"] == result["initial_grid"], "Source mismatch")
    pixels = [(r, c) for r, row in enumerate(grid) for c, label in enumerate(row) if label]
    initial = tuple(grid[r][c] for r, c in pixels)
    index = {p: i for i, p in enumerate(pixels)}
    # Build with offsets, independently of Task 2's pair-distance construction.
    offsets = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)
               if (dr > 0 or (dr == 0 and dc > 0))]
    edges = [(i, index[(r + dr, c + dc)], math.hypot(dr, dc))
             for i, (r, c) in enumerate(pixels) for dr, dc in offsets
             if (r + dr, c + dc) in index]
    require(len(pixels) == result["variable_count"] == 8, "Variable count mismatch")
    require(len(edges) == result["edge_count"] == 28, "Edge count mismatch")
    require(len({(i, j) for i, j, _ in edges}) == len(edges), "Duplicate edges")
    bound = sum(10 / distance for _, _, distance in edges)
    require(math.isclose(bound, result["strong_fidelity_bound"], abs_tol=1e-9), "Bound mismatch")
    require(result["attribute_term_F"] == 0 and result["distance_exponent_eta"] == 1,
            "Unexpected pairwise parameters")

    def pairwise(labels):
        return sum(COST[labels[i]][labels[j] - 1] / distance for i, j, distance in edges)

    runs = result["runs"]
    require([(r["setting"], r["lambda"], r["binary_enabled"]) for r in runs] ==
            [("unary_only", 1, False), ("strong_fidelity", math.ceil(bound) + 1, True),
             ("weaker_fidelity", 1, True)], "Unexpected run settings")
    # Independently enumerate once, checking each setting with scalar arithmetic.
    minima = [math.inf] * len(runs)
    counts = [0] * len(runs)
    first = [None] * len(runs)
    for labels in itertools.product(range(1, 6), repeat=len(pixels)):
        changes = sum(a != b for a, b in zip(labels, initial))
        spatial = pairwise(labels)
        for k, run in enumerate(runs):
            energy = run["lambda"] * changes + (spatial if run["binary_enabled"] else 0)
            if energy < minima[k] - 1e-9:
                minima[k], counts[k], first[k] = energy, 1, labels
            elif abs(energy - minima[k]) <= 1e-9:
                counts[k] += 1
    checks = []
    for k, run in enumerate(runs):
        output = run["returned_grid"]
        with (ROOT / "pannuke_task2" / f"{run['setting']}_labels.csv").open() as stream:
            require(output == [list(map(int, row)) for row in csv.reader(stream)], "Output CSV mismatch")
        require(len(output) == len(grid) and all(len(a) == len(b) for a, b in zip(output, grid)),
                "Output dimensions changed")
        require(all(output[r][c] == 0 for r, row in enumerate(grid)
                    for c, label in enumerate(row) if label == 0), "Background changed")
        labels = tuple(output[r][c] for r, c in pixels)
        require(all(label in COST for label in labels), "Invalid returned class")
        changes = sum(a != b for a, b in zip(labels, initial))
        unary = run["lambda"] * changes
        binary = pairwise(labels) if run["binary_enabled"] else 0
        initial_energy = pairwise(initial) if run["binary_enabled"] else 0
        for value, reported in [(unary, run["returned_unary"]), (binary, run["returned_pairwise"]),
                                (unary + binary, run["returned_objective"]),
                                (initial_energy, run["initial_objective"]),
                                (minima[k], run["returned_objective"])]:
            require(math.isclose(value, reported, rel_tol=0, abs_tol=1e-9), "Objective mismatch")
        require(changes == run["changed_pixels"], "Changed count mismatch")
        require(counts[k] == run["optimal_assignment_count"], "Tie count mismatch")
        require(labels == first[k], "Tie representative mismatch")
        require(run["assignments_evaluated"] == 5 ** len(pixels), "Enumeration count mismatch")
        if run["setting"] == "strong_fidelity":
            require(run["lambda"] > bound and labels == initial, "Fidelity check failed")
        checks.append({"setting": run["setting"], "verified_objective": unary + binary,
                       "optimal_assignment_count": counts[k], "passed": True})
    (HERE / "verification.json").write_text(json.dumps({
        "source_image": metadata["source_image"], "variable_count": len(pixels),
        "unique_edges": len(edges), "assignments_checked_per_setting": 5 ** len(pixels),
        "method": "independent scalar exhaustive enumeration; absolute tolerance 1e-9",
        "checks": checks, "all_checks_passed": True,
    }, indent=2) + "\n")
    print("PASS: source, domains, background, edges, costs, optima, ties, and strong fidelity")


if __name__ == "__main__":
    main()
