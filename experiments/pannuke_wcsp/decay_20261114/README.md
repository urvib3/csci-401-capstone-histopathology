# PanNuke hierarchical sibling decay

The original V1 objective and optimizer are reused. A candidate supplies six
sibling weights, one per level from pixels to 32×32. Each is applied to the
corresponding original table before the solver's division by four. Parent-child
mismatch stays at 0.10. V1 uses only pixel fidelity. The V2 checks add block
fidelity at 0.30.

Run from this directory with the repository Python environment:

```sh
../../../.venv/bin/python run.py dev
../../../.venv/bin/python run.py audit
../../../.venv/bin/python run.py gate
../../../.venv/bin/python run.py archive
../../../.venv/bin/python run.py prepare
../../../.venv/bin/python run.py evaluate
../../../.venv/bin/python validate.py
../../../.venv/bin/python figures.py
```

- `dev` compares block-only, all-level and pixel-only schedules, two coarse
  boosts, lower uniform block weights, matched Potts block schedules, and two
  V2 conditions. It saves the development results and freezes the finalists.
- `audit` checks equality against the existing V1 outputs on the original
  comparison set. It does not use those outcomes for model selection.
- `prepare` samples new fold-2 patch IDs after excluding documented prior
  uses, then reuses the previous full-patch HoVer-Net inference path. The sample
  is patch-disjoint from those identified uses. Slide and patient independence
  and prior checkpoint exposure are unverified.
- `evaluate` reads only frozen choices and compares them on the new sample.
  Confusion matrices and the per-patch metrics support paired analysis.
  Predictions and inference caches are local outputs.

The anchored decay sweep holds 32×32 beta at 0.30. It weakens finer levels
relative to V1. The separate coarse-boost runs directly strengthen 16×16 and
32×32 costs relative to V1.

The final comparison includes the best nonuniform V1 decay schedule and
its matched Potts control even when the overall hybrid choice is a different
formulation. This diagnostic was added after inspecting development results
and before comparing any new-sample model predictions.

The primary metric averages all five biological classes. The new sample has
only one Dead-positive patch. As a secondary sensitivity check specified
before evaluation, `validate.py` computes paired intervals for the other four
biological classes. This does not change the frozen model selection.

An optional dev-only check tests whether accepting WCSP edits only when the
HoVer-Net top-two probability margin is small improves over ungated finalists.
`run.py gate` scores four thresholds for each finalist and writes `gate.json`.
A gated candidate must clear the same development margin to advance.
The gate is an output rule applied after ICM. The local-minimum audit applies
to the ICM assignment before that rule.

The `archive` stage checks the frozen decay schedule against previously
inspected predictions and includes the matched Potts control. The archived
results are exploratory. They were computed after freezing choices and do
not feed the new-sample selection.

- [Development sweep and exact weights](development.csv)
- [Frozen settings](frozen.json)
- [Patch manifest and exclusions](split.json)
- [Archived sample results](archived.json)
- [New-sample results and paired comparisons](evaluation.json)
- [Sensitivity excluding Dead](sensitivity.json)
- [New-sample comparison figure](evaluation.png)
- The optional example montage and regenerated bootstrap draws are local artifacts.
- [Validation receipt](validation.json)
