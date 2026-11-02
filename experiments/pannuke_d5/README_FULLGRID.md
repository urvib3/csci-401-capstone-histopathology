# PanNuke: one-stage WCSP (current experiment)

`run_fullgrid.py` jointly assigns every pixel one of six labels: background (0) or a nucleus type (1–5). There is one optimization, with no intermediate active/background mask to freeze. After solving, `active_mask = refined_labels > 0`.

## Probability-based unary costs (current default)

HoVer-Net computes probability scores, but the saved instance maps and classification CSV do not preserve the per-pixel evidence needed here. The installed TIAToolbox `infer_batch` applies an argmax to the type head. `export_hovernet_probabilities.py` instead calls the model forward pass and saves both maps before that information is discarded. It uses only RGB images and the local checkpoint, not ground truth.

For each pixel, let `pN = P(nucleus)` from the NP head, and `pType[k] = P(type k | nucleus)` from the biological TP logits normalized over classes 1–5. Then:

- `U(background) = -log(1 - pN)`
- `U(type k) = -log(pN * pType[k])`

Subtract the smallest unary cost at each pixel so its cheapest candidate costs zero; all cost differences are preserved. Zero probabilities are clipped at 1e-8 for numerical stability. These are actual softmax model scores, not a guarantee of calibrated probabilities.

There is **no fixed unary penalty of 2** in this mode. For example, with P(nucleus)=0.1 and P(type|nucleus)=0.8, switching background to that type has a relative penalty `log(0.9/0.08) = 2.42`; with P(nucleus)=0.9 it instead favors that type over background by `log(0.72/0.1) = 1.97`.

The calibrated biological transition matrix remains an optional extra prior (`--add-type-transition-prior`). It applies only to original nucleus-type → nucleus-type changes; it never adds a fixed background transition cost. Probability costs alone are the default.

## Historical transition-table mode

Each pixel gets a six-value vector from its **original HoVer-Net class's row** below. Rows are original classes; columns are candidate classes. The reference row stays fixed throughout optimization.

| Original → candidate | Background | Neo | Inflam | Connective | Dead | Non-neo epithelial |
|---|---:|---:|---:|---:|---:|---:|
| Background | 0 | 2.000 | 2.000 | 2.000 | 2.000 | 2.000 |
| Neoplastic | 2.000 | 0 | 0.937 | 0.711 | 1.402 | 1.013 |
| Inflammatory | 2.000 | 1.155 | 0 | 0.710 | 1.681 | 1.681 |
| Connective | 2.000 | 0.877 | 0.794 | 0 | 1.705 | 1.477 |
| Dead | 2.000 | 0.631 | 0.154 | 0.235 | 0 | 1.123 |
| Non-neoplastic epithelial | 2.000 | 0.829 | 0.728 | 0.814 | 1.341 | 0 |

- Keeping the original label costs **0**.
- Background ↔ any nucleus type costs **2**, retaining the previous untuned foreground/background penalty. Unknown background pixels have no saved type confidence; no image evidence is introduced by these costs.
- Type → type costs use the calibrated directed matrix from **100 separate patches / 1,054 nuclei**, excluding development, frozen evaluation and the original example. Rarely supported corrections cost more. Rule: `log(1 + (C[a,a]+1)/(C[a,b]+1))`, diagonal zero, with mean biological off-diagonal penalty normalized to 1. Calibration is patch-disjoint, not verified patient-disjoint.

Exact values: [unary matrix](unified_transition_costs.json). Calibration provenance: [fit.json](unary_calibration/fit.json). This unary matrix is separate from the symmetric adjacency table.

## Pairwise costs and solver

The handwritten five-type adjacency table is unchanged. Add a background row/column: background–background costs 0 and background–nucleus costs **0.25**. Divide each edge cost by pixel distance, using the original 24 neighbors up to 2√2 pixels away.

Minimize **sum of unary transition costs + sum of pairwise costs** using block coordinate descent with ICM. Presence and type can change in the same solve. This is local optimization, not a globally exact solver.

Unlike the two-stage experiment, variables are pixels, not whole nuclei; there is no separate 12-pixel nucleus graph. Pixels of a nucleus can acquire different types. The final class map does not itself establish new instance IDs or whole-cell boundaries. Prediction coverage remains the central 164 × 164 region.

## Run with actual probability maps

Use the existing local checkpoint to export a patch (on Colab, add `--device cuda`):

```bash
python experiments/pannuke_d5/export_hovernet_probabilities.py \
  --indices 748 --output /tmp/pannuke_probabilities

python experiments/pannuke_d5/run_fullgrid.py \
  --source experiments/pannuke_d5/pannuke_preparation \
  --probabilities /tmp/pannuke_probabilities/image_0748/hovernet_probabilities.npz \
  --output /tmp/pannuke_probability_wcsp

python experiments/pannuke_d5/evaluate_fullgrid.py --run /tmp/pannuke_probability_wcsp
```

The exporter streams RGB patches from the local Fold 2 zip. If WSI dependencies are unavailable, `--architecture-source PATH` can point to the installed `tiatoolbox/models/architecture` directory to load only its official inference definitions. Arrays are P(nucleus) `(164,164)` and conditional type probabilities `(5,164,164)`, aligned to `[46:210,46:210]`. If probabilities are absent, the runner stops rather than fabricating them from labels or CSV confidence values. Probability unary-only optimization selects the joint-probability argmax, which can differ from saved post-processed instance labels.

For all 15 development patches:

```bash
python experiments/pannuke_d5/export_hovernet_probabilities.py \
  --selection experiments/pannuke_d5/pannuke_lambda_dev_selection.csv \
  --output /tmp/pannuke_dev_probabilities

python experiments/pannuke_d5/run_unified_dev.py \
  --probability-dir /tmp/pannuke_dev_probabilities \
  --output /tmp/pannuke_probability_dev
```

By default, both runners initialize from the saved post-processed HoVer-Net label map; `--initialization probability_argmax` selects a purely pixel-probability initialization. Neither uses clustering. The development runner reports the raw joint-probability argmax as an additional baseline, then compares WCSP probability costs alone with WCSP probability costs plus the optional type-transition prior. This separates effects of the new probability labeling from refinement itself. It uses the original HoVer-Net labels as initialization. The old table-based experiment remains available with `--unary-mode transitions`. Optimization never receives ground truth.

## Probability-based development evaluation (15 patches)

Actual HoVer-Net probability maps were re-exported from the original cached checkpoint. SHA-256 matches `789b2be8bdaf532627f584a009f83b3ba06e04c0e55c1635252ff12785ef57e8`. The inference-only loader uses unchanged network definitions from installed TIAToolbox source and loads all checkpoint parameters/buffers strictly. Source hashes, alignment and model provenance are saved with each [probability map](hovernet_probability_dev_maps/).

The 15 development patches, original ground-truth crops, and WCSP settings are unchanged. The frozen 20-patch evaluation is not used. No costs were tuned during this run.

| Method | Mean biological mIoU | Δ vs. saved HoVer-Net | Mean binary nucleus IoU | Mean pixel accuracy |
|---|---:|---:|---:|---:|
| Saved HoVer-Net instance labels | 0.474933 | — | 0.746156 | 0.889847 |
| Raw joint-probability argmax, no WCSP | 0.443796 | −0.031137 | 0.754598 | 0.887061 |
| One-stage probability WCSP | **0.475899** | **+0.000965** | 0.752455 | 0.891235 |
| Probability WCSP + calibrated type prior | **0.475922** | **+0.000989** | 0.752476 | 0.891255 |

- **A small average improvement, not a consistent improvement:** each WCSP version improves 6 patches and worsens 9 relative to saved HoVer-Net.
- Probability WCSP makes **2,014 wrong → correct** changes and **1,454 correct → wrong** changes. With the optional type prior: 2,018 corrections and 1,450 new errors.
- **The type prior adds very little:** mIoU increases by only 0.000023 over probability costs alone.
- **Probability maps alone do not explain the class improvement:** their raw six-label argmax is worse than saved post-processed HoVer-Net labels. WCSP raises its mean mIoU from 0.443796 to 0.475899. For binary masks, however, raw probability argmax has higher IoU than WCSP, so segmentation and classification effects differ.
- Alignment checks: raw NP masks versus saved active masks have mean Dice 0.9901; conditional type-map majority matches the saved nucleus type for 234/237 nuclei. These are agreement checks, not ground-truth accuracy scores.

[Summary](unified_probability_dev_trial/summary.json) · [Per-patch scores](unified_probability_dev_trial/per_patch.csv) · [Saved probability maps](hovernet_probability_dev_maps/)

The gain is preliminary development evidence; it does not establish performance on an independent test set. The previous transition-cost scores below remain historical comparisons.

## Pixel-information-flow check: no nucleus labels in initialization

A separate controlled run initializes WCSP directly from each pixel's probability vector, rather than the saved per-nucleus class map. The optimizer uses all 26,896 pixels, six labels, full per-pixel unaries, the same spatial edges and costs, and no calibrated type-transition prior. There is no clustering or nucleus-level averaging. Saved HoVer-Net labels are read only for evaluation in this run.

| Method | Mean biological mIoU | Mean binary nucleus IoU |
|---|---:|---:|
| Saved HoVer-Net | 0.474933 | 0.746156 |
| Pixel-probability argmax | 0.443796 | 0.754598 |
| Pixel-initialized WCSP | 0.447382 | 0.749135 |
| Same probability WCSP, post-processed HoVer-Net initialization | 0.475899 | 0.752455 |

Pixel-initialized WCSP improves its raw probability baseline on 10/15 patches, with mean delta mIoU +0.003586. Against the saved HoVer-Net output, however, only 5/15 improve and 10 worsen; mean delta is -0.027551. It makes 2,349 corrections and 3,037 new errors relative to saved HoVer-Net.

**Interpretation:** full pixel information reaches WCSP in both probability-based variants. Initialization changes the outcome of local ICM search: the pixel-initialized solution has lower energy than the post-processed-initialized solution on 7 patches, higher on 5 and equal on 3. Lower energy is not a guarantee of better accuracy. This experiment does not show that removing HoVer-Net's post-processing improves refinement.

The default retains HoVer-Net initialization because it performed better in these development comparisons. That does not freeze its labels, create nucleus variables, or discard the full pixel probabilities. It is not the centroid clustering considered in Deliverable 3.

[Summary](pixel_information_flow_dev_trial/summary.json) · [Per-patch scores](pixel_information_flow_dev_trial/per_patch.csv)

```bash
python experiments/pannuke_d5/run_unified_dev.py \
  --probability-dir experiments/pannuke_d5/hovernet_probability_dev_maps \
  --initialization probability_argmax --skip-type-prior \
  --output /tmp/pannuke_pixel_information_flow
```

## Original six-channel TP probabilities, pixel-only WCSP

Using the original TP softmax directly, without NP fusion or a type-transition prior, gives mean biological mIoU **0.438640 → 0.440987** after WCSP on the same 15 patches. Saved HoVer-Net remains better at **0.474933**. Mean binary nucleus IoU falls **0.732753 → 0.723412**. Initialization is raw TP argmax, with the existing neighbor costs unchanged. No nucleus averaging or clustering is used. This configuration does not improve the saved baseline.

[Settings and results](original_tp_wcsp_dev_trial/README.md) · [Summary](original_tp_wcsp_dev_trial/summary.json) · [Convergence and raw-baseline audit](original_tp_wcsp_dev_trial/audit.json)

## HoVer-Net initialization and neighbor-strength calibration

Original TP probability unaries were tested from saved HoVer-Net initialization at neighbor strengths 0, 0.1, 0.25, 0.5, 1 and 2 on the 15 development patches, with unchanged HoVer-Net as a separate candidate. The best actual WCSP strength was 2 (mIoU 0.462425), below unchanged HoVer-Net (0.474933), so the frozen overall choice retained HoVer-Net unchanged.

The frozen strength-2 WCSP candidate was also evaluated on 20 new patches excluded from earlier selections. Class mIoU rose **0.555525 → 0.559888**, with 13 patches improving and 7 worsening, but binary nucleus IoU fell **0.770946 → 0.764872**. The setting was not reselected from evaluation results. This is mixed evidence, not a guarantee of equal-or-better refinement. Samples are patch-disjoint within Fold 2; patient independence is not established.

[Calibration and commands](pixel_wcsp_strength_calibration/README.md) · [Frozen evaluation](pixel_wcsp_frozen_evaluation/README.md)

## Extended frozen evaluation: 100 additional patches

The same frozen strength-2 WCSP candidate was evaluated on 100 further patches across 18 tissue types, excluding all earlier selections. No parameters were retuned. Mean biological mIoU changed **0.595110 → 0.595380** (+0.000270), while mean binary nucleus IoU declined **0.743264 → 0.739190**. Class mIoU improved on 44 patches, worsened on 55 and was unchanged on one. The paired patch bootstrap 95% interval for mean mIoU change is **[-0.007761, +0.008223]**. This does not establish a consistent overall benefit.

All 100 runs reached single-pixel local minima with nonincreasing energy. Only Fold 2 was locally available; samples are patch-disjoint from earlier selections, without verified patient independence. Per-class and per-tissue comparisons are in the report; rare-class and small tissue-group findings require caution.

[Full results and reproducible commands](pixel_wcsp_extended_evaluation/README.md) · [Per-class IoU](pixel_wcsp_extended_evaluation/per_class.csv) · [Per-tissue results](pixel_wcsp_extended_evaluation/per_tissue.csv)

## Background-inclusive metric reassessment

The earlier headline biological mIoU excluded background from its class average, although background was already a WCSP label and its errors affected class unions and full pixel accuracy. Saved predictions were re-evaluated without optimization or tuning. On the 100 additional patches, mean patch mIoU over GT-present classes **including background** changes **0.723194 → 0.723400**; **45 patches improve and 55 worsen**. This does not meet a non-degradation target.

Dataset-pooled six-class mIoU is **0.608028 → 0.636522**. This also changes the aggregation from patch means to pooled class intersections/unions; it is not directly comparable with the earlier headline. About 95% of its net gain comes from the Dead class, present in only four patches. Background IoU itself falls **0.945354 → 0.944390**. Fewer background pixels are called nuclei, but more nucleus pixels are missed.

[Background-inclusive reassessment](pixel_wcsp_extended_evaluation/BACKGROUND_REASSESSMENT.md) · [Development reassessment](pixel_wcsp_strength_calibration/BACKGROUND_REASSESSMENT.md) · [Earlier 20-patch reassessment](pixel_wcsp_frozen_evaluation/BACKGROUND_REASSESSMENT.md)

## Historical development results (transition costs, 15 patches)

| Method | Mean biological mIoU | Mean binary nucleus IoU |
|---|---:|---:|
| HoVer-Net | 0.474933 | 0.746156 |
| Unified, flat type costs; same background costs | 0.473998 | 0.745030 |
| Unified, class-dependent type vectors | 0.473998 | 0.745030 |
| Previous two-stage, class-dependent type vectors | 0.432511 | 0.746096 |

The unified experiment improves 3 patches, worsens 5 and leaves 7 unchanged. It makes 34 corrections and 57 new errors. Class-dependent and flat type costs produce identical labels on these patches. Neither improves the HoVer-Net baseline. The two-stage comparison also changes the variable unit and graph, so it does not isolate stage count alone.

[Summary](unified_class_costs_dev_trial/summary.json) · [Per-patch results](unified_class_costs_dev_trial/per_patch.csv)

```bash
python experiments/pannuke_d5/run_unified_dev.py --unary-mode transitions --output /tmp/pannuke_unified_dev
```

The transition-table scores in this historical section predate probability-based costs; they must not be presented as results of the new default. The development runner reads cached original HoVer-Net labels, not the two-stage predictions. Old two-stage scripts and results remain as historical comparisons. BCSS and historical D5 results are unchanged.
