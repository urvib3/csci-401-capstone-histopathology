# Full-512 BCSS result snapshot for Deliverable 7

These compact artifacts document the frozen November 22–23 correction runs.
They contain no raw dataset tiles, probability maps or model checkpoints.

- `corrections_dev.json`: November 23, 15:16 PST snapshot; 34 complete
  configurations on all 185 dev tiles, with partial counts for the others.
  Accuracy is withheld for incomplete populations. The appearance grid was
  still running when exported. Bootstrap intervals recorded on each family's
  best completed setting compare against its named mu-matched control.
- `expanded_dev.json`: completed 17-treatment lambda/mu and convergence study;
  curve checkpoints produce additional report rows, not additional treatments.
- `validation.json`: completed fixed-choice full-validation evaluation;
  all 4,028 tiles accounted for, 3,229 scored and 799 wholly ignored.
- `provenance.json`: model/source hashes, sample IDs, input hashes, effective
  treatments, experiment base commit and the upstream checkout used for D7.
- `presence_prior.json` and `appearance_model.json`: the exact frozen fit-only
  class-set prior and Gaussian appearance parameters/calibration provenance.
- `region_energy_audit.json`: per-tile energy comparison against the matched
  pixel control; records the independent energy-decrease check.
- `region_examples.json`: exact IDs and post-inspection selection rule for
  the median positive and median negative per-tile examples.

The correction snapshot and expanded study use the same 185 frozen full-field
development tiles. The validation table is a different population and must
not be ranked against those development scores. Settings were selected on
development data; the new region/presence candidates do not yet have a
full-validation result. Intervals after development selection are exploratory
and are not adjusted for the number of configurations screened.

Rebuild the data-driven charts from the repository root:

```bash
python deliverables/figures_20261123/make_full512_figures.py
```

To regenerate the illustrative image panel, additionally pass `--cache-root`
pointing to the original experiment checkout containing the fullres input,
asset and prediction caches. Image-based examples are selected after seeing
dev outcomes and are descriptive, not independent evidence.

The running experiments retain their original checkout and frozen sources.
Upstream changed several WCSP solvers after those protocols were frozen.
The copied runners pass their focused tests against this newer checkout,
but exact replay/resume of an existing output root still requires the source
hashes in its original protocol. Use a separate output root for a new solver
revision rather than changing the recorded experiment history.
