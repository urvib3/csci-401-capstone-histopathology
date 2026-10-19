# PanNuke Task 3 — Independent Verification

After Task 2, run from the repository root:

```bash
python3 benchmarks/deliverable_4/pannuke_task3/verify_results.py
```

The verifier uses only the Python standard library and does not import Task 2's implementation. It independently builds neighbors using offsets, transcribes the supplied cost matrix, and enumerates all 390,625 assignments for each of the three settings.

It checks the source labels, eight active variables, 28 unique edges, output CSVs, five-class domains, unchanged background, objective components, changed-pixel counts, fidelity bound, global minima, and optimal-assignment counts. Numerical comparisons use absolute tolerance `1e-9`.

All checks passed. Strong fidelity preserved the initial labels. Weaker fidelity changed three pixels to Connective; this is an objective reduction, not a demonstrated accuracy improvement. `verification.json` records the checks. The submission narrative and before/after grids are in the parent README.
