# Full-patch hierarchical PanNuke WCSP

## Run order

```bash
python experiments/pannuke_wcsp/hierarchy_20261109/run.py prepare
python experiments/pannuke_wcsp/hierarchy_20261109/run.py infer
python -m unittest discover -s experiments/pannuke_wcsp/hierarchy_20261109 -p "test_*.py" -v
python experiments/pannuke_wcsp/hierarchy_20261109/run.py calibrate
python experiments/pannuke_wcsp/hierarchy_20261109/run.py evaluate
python experiments/pannuke_wcsp/hierarchy_20261109/report.py figures
python experiments/pannuke_wcsp/hierarchy_20261109/report.py text_artifacts
python experiments/pannuke_wcsp/hierarchy_20261109/report.py pdfs
python experiments/pannuke_wcsp/hierarchy_20261109/validate.py
```

- Dependencies: Python 3.12, NumPy, Numba, Torch, SciPy, Matplotlib; ReportLab for PDFs. Exact runtime versions are in inference.json.
- Existing split/config/results refuse overwrite. For reproduction use a fresh copy of this experiment directory, retaining original manifests/configs for verification; do not overwrite this run.
- Input archive, checkpoint and official model source are resolved through the previous D5 loader. Exact checkpoint/source hashes are in inference.json.
- cache/ contains local full RGB, GT and probabilities; never commit raw data/checkpoints/environments. Compact semantic predictions and metadata are reproducibility outputs.
- Internal guide is ignored by this experiment .gitignore; report.pdf is the detailed public teammate report.

## Energy

`E = sum pixel U + gamma sum block U + (sibling/4) sum C_level(a,b) + parent sum [a != b]`.

- U = -log(max(probability,1e-12)), shifted to minimum zero per variable.
- Pixel probabilities are raw six-channel TP softmax. Block probabilities are arithmetic means over underlying pixels, not iterative hard-label votes.
- gamma=0 in V1; gamma chosen from [.1,.3,1] using only development patches in V2.
- Shared sibling/parent pair chosen on V1 development pooled biological mIoU from (.1,.03), (.3,.1), (.6,.3); held fixed for V2. This favors V1 calibration; a limitation.
- Four-connected same-level spatial edges, including across parent boundaries. This follows immediate adjacency in the hierarchy PDF rather than the historical flat 24-neighbor graph.
- Every non-top node has one parent. Root is a container, no variable or energy term.
- Each level: GT plurality labels for training blocks; tie -> lowest class ID. Learned cost = positive part of negative log prevalence-adjusted contact ratio, add-one smoothing, zero diagonal and unit off-diagonal mean.
- Prior is the previously user-prescribed revised five-class matrix, normalized separately; Background entries come from that level learned table. Blend 80% learned / 20% prior and renormalize.
- Biological reference rules are retained, not rediscovered; separate learned tables and per-node weighting provide scale adaptation. New coarse-scale biological priors are not independently validated.
- Dead has no dominant-class contact observations at 32x32 in the fitting set. Smoothing and the prior keep costs finite, but do not create empirical support.
- Alternating forward/reverse sweeps use strict energy decrease, retaining labels on ties; up to 100 full rounds. Edge-list energy and conditional-cost audit independently check the CSR solver.

## Data and metrics

- Full-patch inference reflect-pads 46 pixels and uses four 256-input/164-output tiles at output origins (0,0), (0,92), (92,0), (92,92), averaging overlapping probabilities. No original pixel is cropped or resized.
- This changes the inference protocol from earlier central-only reports; compare methods only within this paired run. Original surrounding slide context is unavailable. Native GT biological channels map to IDs 1-5; highest ID wins overlaps; Background means no biological annotation.
- Reuse the prior 200 fitting patches; exclude identifiable previous use from the 12 dev and 100 test patches. All selections use GT class/tissue metadata, not prediction outcomes. Exact IDs and reasons are saved.
- Pooled biological mIoU always averages classes 1..5; six-class averages 0..5. Zero union -> zero IoU. Per-patch metrics use the same fixed denominator.
- Bootstrap resamples paired whole patches 5,000 times and pools each sample before IoU. It is not a patient/slide bootstrap.
- Boundary diagnostics: within two four-connected dilation steps of any GT label transition. Confidence uses top probability and top-minus-runner-up margin.
- Version 2 runs if V1 fails pooled improvement OR any patch worsens. This conditional choice is disclosed; no evaluation-outcome tuning.

## Files

- split.json, *_manifest.csv, exclusions.json, coverage.json: sampling and coverage.
- tables.json, table_level_*.csv: counts, priors, learned/final scale tables.
- frozen_config.json: shared parameters and dev selection.
- version_*/results.json, confusions.npz, per_patch.csv, patch_*.json/npz: metrics, audits, pixel/block labels.
- figures/: presentation-ready plots and paired examples.
- report.pdf: detailed method/results/limitations; internal_guide.pdf: private review aid.
- Deferred controls have not been run. Nothing automatically committed or pushed.

## Upstream prior-use correction

- After fetching the team updates, four comparison IDs (908, 2216, 2330, 2354) were found in the upstream beta-development sweep. This run was selected before that sweep was locally available.
- The original 100-patch predictions, scores and frozen parameters are preserved. Treat those results as exploratory rather than completely unused-test evidence. No fitting or current development IDs overlap that sweep.
- See upstream_overlap_audit.json for provenance. Future splits must include the newly merged exclusions.
