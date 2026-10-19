# PanNuke Task 2 — Tiny WCSP

Run from the repository root with Python and NumPy:

```bash
python3 benchmarks/deliverable_4/pannuke_task2/run_wcsp.py
```

The script reads Task 1's CSV, checks it against the metadata and NPY, and uses only its eight nonzero pixels as variables. Every variable has all five cell classes available. It enumerates all 390,625 assignments per setting; no external solver is required.

Costs are `lambda * changed_pixels` plus the handout's base pairwise costs divided by distance. The attribute term is zero and the distance exponent is one. All five neighbor types are included, with each active pair counted once. Background and outside-crop interactions are omitted. Costs are not scaled or rounded.

- Unary-only: lambda 1, pairwise disabled; optimum 0, no changes.
- Strong fidelity: lambda 193, above bound 191.330290; optimum 22.804398, no changes.
- Weaker fidelity: lambda 1, pairwise enabled; optimum 3, three changes to Connective.

All three optima are unique within tolerance `1e-9`. `results.json` contains parameters, costs, timing, edge counts, and returned grids. `results.csv` contains the run summary; `*_labels.csv` contains the returned arrays. Rerunning overwrites these outputs and updates runtimes. The submission narrative is in the parent README.
