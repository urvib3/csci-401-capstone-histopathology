# Larger replication of frozen PanNuke type-set WCSP

The model, nucleus masks, type unaries, fitted frequencies, and type-set
constraint are the same as in the previous study. No hyperparameters are
selected on these patches.

1. **Local evidence.** HoVer-Net produces a type-probability vector at each
   pixel. Once its detection heads define a nucleus mask, pixels inside
   that mask vote for the nucleus type. A type with more votes has a lower
   unary cost for that nucleus.
2. **Patch context.** Each proposed labeling produces a set of types
   present somewhere in the patch. The WCSP adds a cost based on how
   common that set was in 200 fitting patches with roughly the same
   number of nuclei. It also charges for proposed types that have weak
   support in the model's pooled probabilities. A rare type can still win
   if its evidence is strong enough.
3. **Optimization.** For each of the 31 possible nonempty sets of five
   types, solve the minimum-cost assignment that uses every type in that
   set. Add the two patch penalties and pick the cheapest solution
   across sets. Empty detection masks stay empty.

For example, several nuclei might have strong votes for Connective and
one has nearly tied votes for Connective and Dead. Choosing Dead for the
ambiguous nucleus both changes its local cost and introduces Dead into
the set of types present. Whether that happens depends on the summed
cost. No ground-truth type is available to the solver for that patch.

## Evaluation

- **Primary:** 300 randomly sampled patches outside every documented
  earlier fitting, development, and comparison set. Report pooled
  biological mIoU and 95% paired patch-bootstrap intervals.
- **Dead-positive stress set:** all other 21 eligible patches containing
  annotated Dead nuclei. Their results characterize a targeted subset
  and are reported separately from the primary sample.
- **Sensitivity:** report changes over the four other biological classes,
  leave the original five-class result as the primary metric, and
  report per-class changes for both samples.
- **Controls:** Raw TP uses the per-pixel argmax. Fixed-mask vote uses
  identical nucleus masks and local type evidence as WCSP without the
  patch penalties. The latter isolates the added value of the constraint.
- Patch IDs are disjoint from documented prior uses. Whether patches
  share slides or patients and whether they are independent of HoVer-Net
  checkpoint training are unverified.

## Reproduction

From this directory with the repository Python environment:

```sh
../../../.venv/bin/python prepare_scale.py
../../../.venv/bin/python prepare_scale.py --worker 0
../../../.venv/bin/python prepare_scale.py --worker 1
../../../.venv/bin/python prepare_scale.py --worker 2
../../../.venv/bin/python evaluate.py
../../../.venv/bin/python figures.py
```

Workers may run concurrently. Inference caches and per-nucleus assignments
are kept locally. The [split](split.json) and [protocol](protocol.json)
record the IDs and hashes of the exact frozen inputs.

The recorded run used CPU inference for completed work and Apple GPU
inference for the remaining patches. The same model, tile layout, fusion,
and postprocessing ran on both paths. On a random 12-patch reference
check, both devices produced identical Raw TP pixel labels, nucleus
masks, and frozen WCSP types. The maximum floating-point probability
difference was approximately 8.3 × 10⁻⁶. Run `prepare_scale.py --gpu`
after preparation to use the GPU for every pending patch instead of the
three CPU worker commands.

- [Per-patch inference backends](inference_backends.json)
- [Reference device comparison](accelerator_audit.json)
- [WCSP label equivalence check](accelerator_validation.json)
- [Earlier results reproduced from saved inputs](archived_reproduction.json)
- [Paired results and per-class statistics](evaluation.json)
- [Random sample per-patch scores](primary_patches.csv)
- [Dead-positive per-patch scores](dead_stress_patches.csv)
- [Comparison figure](comparison.png)
