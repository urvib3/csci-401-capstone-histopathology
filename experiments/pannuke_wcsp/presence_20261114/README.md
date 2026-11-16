# PanNuke type-set, count and Raw TP component WCSP

The study evaluates three different constraints on Raw TP predictions from
the same frozen HoVer-Net checkpoint. Its 112 development patches are the
entire previous instance study's development and comparison sets. Its
evaluation uses 100 different patch IDs, excluded from all documented
fitting and development sets. Patch-level intervals use 5,000 paired
bootstrap replicates. Patient and checkpoint independence are unverified.

| Constraint | Variables and objective | Inputs |
|---|---|---|
| Instance type set | Add a penalty for the set of labels present anywhere in a patch. Estimate probabilities on the existing 200 ground-truth fitting patches, conditioned on number of nuclei, with shrinkage and smoothing. An optional evidence term charges unsupported types. | Full-patch HoVer-Net masks and TP probabilities |
| Instance type counts | Add a convex penalty for the difference between class counts and the expected counts from pooled TP probabilities, optionally blended with the fitted global type frequencies. | Same instance masks and TP probabilities |
| Raw TP component | Use a common auxiliary type for each connected raw foreground component. Add a soft penalty when one of its pixels differs from that common type while retaining each pixel's negative log probability. | Raw TP probabilities only |

The type-set solver enumerates every possible set of five types and solves
for the best assignment at each set exactly. The count solver uses ordered
marginal penalties for taking the next slot of each type and solves the
resulting assignment exactly. The component solver enumerates the five
possible component types, then picks each pixel independently conditional
on that component type. It preserves the raw foreground mask.

The fixed-mask foreground-vote control is also the optimum of the
unary-only WCSP with one type per HoVer-Net mask. In pixel terminology,
this enforces hard type agreement inside each detected mask. The previous
Potts solution is included as a pairwise control. Type-set and count
optimization preserve the same masks as these controls, so their effects
can be measured separately.

The answer-key diagnostic in the previous instance study used the correct
ground-truth type for each matched detection. Ground truth is used here
for fitting priors, selecting settings on development data, and scoring
the frozen solutions. Final predictions are generated from model outputs
and frozen costs.

Run with the repository Python environment:

```sh
../../../.venv/bin/python prepare_fresh.py
../../../.venv/bin/python run.py dev
../../../.venv/bin/python raw_star.py dev
../../../.venv/bin/python run.py final
../../../.venv/bin/python raw_star.py final
../../../.venv/bin/python validate.py
```

A secondary sensitivity analysis excludes Dead because the fresh sample contains
only six Dead nuclei across three patches. The primary metric still uses all
five biological types. This sensitivity analysis does not affect model selection.

- [Development grid](development.json) and [frozen instance settings](frozen.json)
- [Instance results and paired comparisons](evaluation.json)
- [Raw TP-only results](raw_only_evaluation.json)
- [Rare-class sensitivity](sensitivity.json)
- [Comparison figure](comparison.png) and [validation receipt](validation.json)
