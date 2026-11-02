# PanNuke: learned pixel compatibility costs, fixed beta 0.30

This standalone experiment asks whether a GT-trained spatial incompatibility table plus four fixed soft biological preferences improves **raw six-channel HoVer-Net TP argmax**. It imports no earlier experiment code. Only two evaluated methods: raw TP argmax and the frozen WCSP at beta 0.30. Beta zero is an equivalence test, not a model-selection variant. No sweep or test-based tuning.

## Environment and reproduction

Run from the repository root using Python 3.12 and the installed `.venv`. Required: NumPy, Numba, Torch, TIAToolbox (source/metadata), PyYAML. Tests use stdlib unittest; pytest is unnecessary. Matplotlib generates scientific figures. ReportLab is required only for PDF generation and should be installed in the active Python environment. Exact experiment versions and checkpoint SHA256 are in `frozen_config.json`. In this run TIAToolbox optional whole-slide dependencies are bypassed: `run.py` compiles only unchanged official architecture/layer/forward definitions from installed sources and strictly loads every checkpoint parameter/buffer. This is not a substitute architecture. Source hashes are recorded.

Local prerequisites (never commit/push):

- `benchmarks/pannuke/data/fold_2.zip` (official 2,523-patch Fold 2).
- `benchmarks/pannuke/data/cache/models/hovernet_fast-pannuke.pth`.
- `.venv/lib/python3.12/site-packages/tiatoolbox`.

The archive contains float64 RGB and masks (~12 GB uncompressed). The pipeline streams each member forward once, keeping compact 164x164 semantic maps and 20 RGB crops rather than extracting it. Cache is ignored by Git. Raw TP logits/probabilities are retained locally in `cache/`; labels, audit records and compact confusion matrices are separate reproducibility outputs.

Exact commands, in order:

```bash
.venv/bin/python experiments/pannuke_wcsp/pannuke_d5/run.py prepare
.venv/bin/python -m unittest discover -s experiments/pannuke_wcsp/pannuke_d5 -v
.venv/bin/python experiments/pannuke_wcsp/pannuke_d5/run.py freeze
.venv/bin/python experiments/pannuke_wcsp/pannuke_d5/run.py evaluate
.venv/bin/python -m unittest discover -s experiments/pannuke_wcsp/pannuke_d5 -v
.venv/bin/python experiments/pannuke_wcsp/pannuke_d5/make_figures.py
# Use a Python with ReportLab for the report:
python experiments/pannuke_wcsp/pannuke_d5/make_report.py
```

Prepare/freeze/evaluate refuse to overwrite their corresponding frozen/final outputs. For an exact frozen reproduction, use `reproduce.py` below. It copies the frozen split/configuration, rebuilds its input cache, recomputes the training table to verify equality, then evaluates the same two methods. Do not run fresh split selection: the existing experiment itself would now appear in the prior-use scan. Do not delete this run's manifests to select another test set. Metadata paths are repository-relative, and scripts locate the repository root from the data directory.

## Pixel variables and provenance

All 26,896 output pixels have labels 0..5, including background. The channel order is read and asserted from installed TIAToolbox `pretrained_model.yaml`, entry `hovernet_fast-pannuke`, `architecture.kwargs.nuc_type_dict`: Background, Neoplastic, Inflammatory, Connective, Dead, Non-Neoplastic Epithelial. Raw `model.forward(...)["tp"]` has shape `(1,6,164,164)` and is transposed to `(164,164,6)` before a numerically stable softmax over **all six channels**. Input is RGB 0..255 float32; the official forward divides by 255 internally. NP/HV branches exist in the pretrained model, but their outputs are discarded and never enter WCSP, initialization or scoring. No instance processing, voting, centroids, or nucleus reconstruction.

Baseline is `argmax(P)`. Unary is `-log(max(P,1e-12))`, minus its pixelwise minimum. All labels remain available. A beta-zero solve must exactly reproduce baseline for every test patch.

## Split and GT

Seed 20261101. Freeze 200 training and 20 test patches before fitting. Seeded GT-only selection seeks at least three test patches per biological class, then tissue diversity; ties use seeded candidate order. Training patches are sampled from the remaining eligible pool. `split.json`, CSV manifests and `coverage.json` give exact IDs, tissue, class pixels and coverage. Test selection never consults predictions. The prevalence of this coverage-enriched sample need not match natural Fold 2 prevalence.

Prior-use exclusions are saved with source paths in `previous_use_exclusions.json`. Explicit prior experiment IDs/selections/evaluation outputs and the first-ten sample are excluded from both splits. The all-patch GT catalog and unused-candidate catalog do not establish use and are excluded from the exclusion scan. Full-fold inference-only caches do not constitute development selection. Unrecorded use cannot be verified. The archive exposes no patient/slide IDs, so patient/slide separation cannot be checked. **Held out means held out from this WCSP table fitting; pretrained HoVer-Net exposure to Fold 2 is unverified.**

GT crop is `[46:210,46:210]`. Biological annotation channels 0..4 map to semantic IDs 1..5; background is no positive biological annotation. This ordering is also documented in `benchmarks/pannuke/data/sample_fold2/README.md`. On overlap the highest biological class ID wins (ascending channels overwrite); overlap counts are saved per patch. Instance numbers are used only as positive GT occupancy, never as WCSP variables. GT background channel is not allowed to override a biological annotation.

## Learn and blend the table

The 24-neighbor graph contains all offsets in the 5x5 window except zero, with squared distances 1,2,4,5,8. Each undirected edge is enumerated once using 12 forward offsets. For training GT adjacency, add both orientations; same-class edges add two diagonal counts. Counts are not distance-weighted. Smooth **every** matrix entry by 1. Normalize to joint `J`; marginal `m_a=sum_b J_ab`. Compute `max(0,-log(J_ab/(m_a*m_b)))`, set diagonal to zero, divide by the six-class off-diagonal mean. This measures incompatibility relative to prevalence rather than penalizing rarity alone.

The supplied 5x5 biological prior has off-diagonal mean normalized to 1. Its biological block enters a six-class prior; background row/column come from normalized learned costs. Blend 80% learned + 20% prior, then divide by the full six-class off-diagonal mean. Tables, counts, joint, marginals, raw costs, intermediate blend, and normalization factors are saved in JSON and CSV. `frozen_config.json` records equations and all fixed choices before inference.

Preferences: same-type homophily (zero diagonal); discourage Dead/Non-Neoplastic more than Dead/Tumor or Dead/Immune; discourage Immune/Non-Neoplastic; permit Connective next to Tumor/Non-Neoplastic while discouraging direct Tumor/Non-Neoplastic. They are soft preferences. **Pairwise costs cannot literally enforce stromal barriers, identify tissue architecture, or detect dense immune clusters.**

## Energy and solver

`E=sum_i U_i(y_i)+sum_{undirected (i,j)} 0.30*C[y_i,y_j]/distance(i,j)`; eta=1. Start from raw TP argmax. Deterministic raster ICM accepts only strictly decreasing local energy moves, retaining current labels on equal cost. At most 100 rounds; stop after a complete round with no changes. Independent vectorized full-grid conditional costs check for remaining single-pixel improvements, with explicit numerical audit tolerance 1e-10. Full energy is independently recomputed every round and compared to accumulated local changes. Per-patch output distinguishes no-change stopping from round limit and records maximum residual gain and strict floating-point candidates. ICM is not globally optimal.

## Metrics and uncertainty

Pool confusion counts **before** computing IoUs. Rows are GT, columns predictions. `IoU=TP/(TP+FP+FN)`. Biological mIoU always divides by five (IDs 1..5); six-class mIoU always divides by six (IDs 0..5). Zero union gives IoU=0 for either method; absent classes are explicitly reported. A GT-absent class with predictions still incurs false positives. Bootstrap resamples the same 20 patch IDs for both methods 5,000 times (seed 20261102), pools sampled confusion matrices, and computes paired biological mIoU differences; percentile 95% CI. It assumes exchangeable patches, not verified independent patients.

`results.json` contains pooled metrics, all patch metrics, fixes/breaks/different-wrong, separate CPU inference and solver times, and bootstrap replicates. `confusions.npz` contains per-patch 6x6 confusion matrices, the baseline-to-WCSP transition matrix, and GT x baseline x WCSP counts. `predictions/` contains aligned GT/baseline/WCSP label arrays and solver audits. Inference timing covers model forward, transfer and softmax; model setup is separate. Solver timing includes independent energy/local-minimum audits and excludes the explicit synthetic warmup and beta-zero validation. Any additional layout-specific JIT specialization inside a timed real solve is included; timings should not be interpreted as pure optimized-kernel speed.

## Figures, report and review deck

- Generate figures and the numeric summary from saved outputs; no inference or optimization is rerun.
- The input validator checks raw RGB values and exact GT crop alignment.
- Install ReportLab in the active environment to generate the PDF.

```bash
python experiments/pannuke_wcsp/pannuke_d5/validate_inputs.py
python experiments/pannuke_wcsp/pannuke_d5/make_figures.py
python experiments/pannuke_wcsp/pannuke_d5/make_deliverable_figures.py
python experiments/pannuke_wcsp/pannuke_d5/make_summary.py
python experiments/pannuke_wcsp/pannuke_d5/make_report.py
```

- `RESULTS.md`: numerical results and solver audits.
- `report.pdf`: three-page illustrated experiment report.
- `review_deck.pptx`: editable review slides, including historical experiments with their original baselines and metric definitions.
- Optional deck-generation source: `deck_builder.mjs` and `make_deck.mjs`; requires the `@oai/artifact-tool` JavaScript package in the Node.js environment. The exported deck is already included.

```bash
node experiments/pannuke_wcsp/pannuke_d5/make_deck.mjs
```

- Deck export refuses to overwrite an existing file; regenerate in the new reproduction directory.
- Optional PDF preview with Poppler:

```bash
mkdir -p experiments/pannuke_wcsp/pannuke_d5/.build/pdf_previews
pdftoppm -r 120 -png experiments/pannuke_wcsp/pannuke_d5/report.pdf experiments/pannuke_wcsp/pannuke_d5/.build/pdf_previews/page
```

## Published metadata

- File references use repository-relative paths; no personal machine paths are required.
- After the experiment directory was reorganized, repository discovery and path serialization were updated for portability.
- Original evaluation configuration and source hashes are retained in `frozen_config.json`; the published configuration has a separate checksum.
- These metadata/path changes do not change the split, learned table, model settings, solver, predictions or metrics. Predictions were not rerun.
- Keep datasets, checkpoints, environments and large probability/logit caches local; they are not required in version control.

## Exact frozen reproduction (no new split)

```bash
.venv/bin/python experiments/pannuke_wcsp/pannuke_d5/reproduce.py --output experiments/pannuke_wcsp/pannuke_d5_reproduction
```

The output must not exist. `--prepare-only` rebuilds the fixed inputs and independently checks the learned table without running inference. The normal command runs exactly the original raw-TP baseline and fixed beta-0.30 WCSP. It preserves the original exclusion snapshot and all 220 patch IDs. Then run the tests and artifact commands above with `pannuke_pixel_learned_costs_reproduction` in place of `pannuke_pixel_learned_costs`. Reproduction requires the exact archived input/checkpoint/source hashes and installed dependency versions recorded in the frozen configuration.

Transition matrices use baseline labels as rows and WCSP labels as columns. The GT × baseline × WCSP tensor in `confusions.npz` uses that axis order.
