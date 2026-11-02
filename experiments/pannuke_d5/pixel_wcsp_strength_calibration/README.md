# Pixel WCSP: HoVer-Net initialization and neighbor-strength calibration

All pixels, including background, remain six-label variables. Unaries use the original six-channel TP softmax: `-log(max(P, 1e-8))`, minus a per-pixel constant. Initialization uses saved postprocessed HoVer-Net labels, with no fixed labels, nucleus voting or clustering in WCSP. The existing biological compatibility table and background/type cost 0.25 remain fixed. Only their shared neighbor strength changes; each edge is weighted by inverse distance across the same 24-neighbor graph.

## Development selection (15 patches)

| Method | Mean biological mIoU | Mean nucleus IoU |
|---|---:|---:|
| unchanged_hovernet | 0.474933 | 0.746156 |
| wcsp_0 | 0.438640 | 0.732753 |
| wcsp_0.1 | 0.441756 | 0.731099 |
| wcsp_0.25 | 0.444455 | 0.729999 |
| wcsp_0.5 | 0.445890 | 0.729344 |
| wcsp_1 | 0.451146 | 0.727594 |
| wcsp_2 | 0.462425 | 0.720349 |

The selection rule maximizes mean patch biological mIoU on development data; exact ties prefer unchanged HoVer-Net, then the lower strength. The best tested actual WCSP strength is 2, but it does not beat unchanged HoVer-Net. The frozen selected method is therefore unchanged HoVer-Net. This is a conservative baseline selection, not evidence that WCSP improves segmentation. The best strength lies at the upper edge of the tested range; no optimum over all possible strengths is claimed.

Zero neighbor strength is unary-only inference and can change saved HoVer-Net predictions. It is distinct from the unchanged baseline. The six-channel TP head and postprocessed HoVer-Net also differ in how they use the network heads and instance processing; this calibration does not isolate neighbor costs from those differences.

## Evaluation protocol

The selected method and best actual WCSP candidate are frozen in `frozen_config.json` before evaluation metrics are read. Twenty new patches are sampled with seed 20261026, excluding the earlier 15 development, 100 calibration and 20 historical evaluation patches plus image 748. Selection is uniform among remaining catalogue entries with saved predictions. These are patch-disjoint samples from the same Fold 2, not a verified patient-disjoint test or an external validation set. No setting is chosen from these evaluation outcomes.

Ground truth is used for development selection and subsequent evaluation, never passed into the WCSP solver. The solver uses block coordinate descent with ICM (20 rounds maximum), so global optimality and equal-or-better ground-truth accuracy are not guaranteed. All 90 development candidate runs have nonincreasing energy and reach a single-pixel local minimum.

Metrics use the established project definition: each patch averages IoU over biological classes present in its ground truth; the reported result averages those patch values. Nucleus IoU combines all five nucleus types against background.

[Development summary](summary.json) · [Frozen configuration](frozen_config.json) · [Evaluation results](../pixel_wcsp_frozen_evaluation/summary.json)

```sh
PYTHONPATH=/private/tmp/pannuke-eval-runtime python3 experiments/pannuke_d5/calibrate_pixel_wcsp.py calibrate --selection experiments/pannuke_d5/pannuke_lambda_dev_selection.csv --prepared experiments/pannuke_d5/two_stage_dev_trial --maps experiments/pannuke_d5/hovernet_raw_head_dev_maps --output /tmp/pannuke_strength_calibration
PYTHONPATH=/private/tmp/pannuke-eval-runtime python3 experiments/pannuke_d5/calibrate_pixel_wcsp.py evaluate --selection experiments/pannuke_d5/pixel_wcsp_new_evaluation_inputs/selection.csv --prepared experiments/pannuke_d5/pixel_wcsp_new_evaluation_inputs --maps experiments/pannuke_d5/pixel_wcsp_new_evaluation_maps --frozen-config /tmp/pannuke_strength_calibration/frozen_config.json --output /tmp/pannuke_strength_evaluation
```

## Frozen evaluation result

On the 20 new patches, the frozen strength-2 WCSP candidate gives class mIoU **0.555525 → 0.559888**, but nucleus IoU falls **0.770946 → 0.764872**. Class mIoU improves on 13 patches and worsens on 7. This small positive evaluation result does not undo the negative development result or prove reliable non-degradation. No setting is reselected after evaluation. [Full evaluation](../pixel_wcsp_frozen_evaluation/README.md).
