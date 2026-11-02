# Frozen pixel WCSP: additional-patch evaluation

100 additional patches across 18 tissue types. One six-label pixel optimization; no binary first stage or nucleus-level WCSP. Full original TP probabilities supply unary costs, saved HoVer-Net labels initialize local search, and neighbor strength remains frozen at 2. No evaluation labels are passed to optimization or used to change settings.

| Method | Mean class mIoU | Nucleus/background IoU | Pixel accuracy |
|---|---:|---:|---:|
| unchanged_hovernet | 0.595110 | 0.743264 | 0.937902 |
| frozen_wcsp | 0.595380 | 0.739190 | 0.938280 |

Class mIoU improves on 44 patches and worsens on 55. Mean change: +0.000270; paired patch bootstrap 95% interval [-0.007761, +0.008223]. It corrects 17,953 pixels and introduces 16,935 new errors.

Headline class mIoU averages biological classes present in each GT patch, then averages patches, matching prior project reports. Binary nucleus/background IoU is an evaluation of those same six-class predictions after merging biological classes, not a pipeline stage.

## Per-class IoU

This table pools pixel intersections/unions across all patches, including false positives where a class is absent. It is not the averaging rule of the headline mIoU.

| Class | GT-present patches | HoVer-Net IoU | WCSP IoU |
|---|---:|---:|---:|
| Background | 100 | 0.945354 | 0.944390 |
| Neoplastic | 50 | 0.744460 | 0.747618 |
| Inflammatory | 30 | 0.559568 | 0.565908 |
| Connective | 69 | 0.517862 | 0.516767 |
| Dead | 4 | 0.330962 | 0.493309 |
| Non-neoplastic epithelial | 20 | 0.549966 | 0.551143 |

## By tissue

Small tissue groups are descriptive only; they are not reliable tissue-specific performance estimates.

| Tissue | Patches | HoVer-Net mIoU | WCSP mIoU | Change |
|---|---:|---:|---:|---:|
| Adrenal_gland | 6 | 0.559464 | 0.546420 | -0.013043 |
| Bile-duct | 4 | 0.467097 | 0.474922 | +0.007825 |
| Bladder | 3 | 0.613863 | 0.636887 | +0.023024 |
| Breast | 31 | 0.604594 | 0.615153 | +0.010559 |
| Cervix | 2 | 0.508206 | 0.496808 | -0.011398 |
| Colon | 21 | 0.605616 | 0.594662 | -0.010954 |
| Esophagus | 6 | 0.579097 | 0.574935 | -0.004163 |
| HeadNeck | 3 | 0.473924 | 0.458671 | -0.015253 |
| Kidney | 3 | 0.652554 | 0.651756 | -0.000798 |
| Liver | 2 | 0.899300 | 0.890195 | -0.009105 |
| Lung | 2 | 0.623503 | 0.652225 | +0.028722 |
| Ovarian | 1 | 0.467216 | 0.448026 | -0.019190 |
| Pancreatic | 3 | 0.605623 | 0.592465 | -0.013158 |
| Prostate | 3 | 0.372719 | 0.384196 | +0.011477 |
| Skin | 3 | 0.591915 | 0.599612 | +0.007697 |
| Stomach | 3 | 0.603596 | 0.603576 | -0.000019 |
| Testis | 3 | 0.753375 | 0.748467 | -0.004907 |
| Thyroid | 1 | 0.818980 | 0.799194 | -0.019786 |

## Limits and selection

Only Fold 2 images and saved predictions were available locally. These patches are excluded from prior development, calibration and evaluation selections, but patient-level independence is not verified. Bootstrap intervals resample patches and do not address possible patient correlations. This is additional patch-level validation within the same fold, not independent external validation.

The development-selected overall method remains unchanged HoVer-Net: all tested WCSP strengths underperformed it on development mIoU. This report separately evaluates the best actual WCSP candidate frozen before the earlier 20-patch evaluation, without using either evaluation set to reselect settings. Equality from retaining the baseline is not evidence of successful WCSP refinement.

All 100 runs have nonincreasing energy; 100 reach a single-pixel local minimum. Local search does not guarantee a global optimum or improved ground-truth accuracy.

[Summary](summary.json) · [Per-class data](per_class.csv) · [Per-tissue data](per_tissue.csv) · [Detailed audit](detailed_report.json)

## Interpretation

The mean class-mIoU gain is only 0.000270 (0.027 percentage points), and its uncertainty interval crosses zero. Of 100 patches, 44 improve, 55 worsen and one is unchanged. Nucleus/background IoU declines. This larger sample does not demonstrate a reliable overall improvement over saved HoVer-Net. Breast mIoU improves on average while colon mIoU declines; other tissue groups are small. The large pooled Dead-class improvement is based on only four GT-positive patches and should not be treated as a robust rare-class result. No setting was retuned after these outcomes.

Reproduce evaluation and reporting:

```sh
PYTHONPATH=/private/tmp/pannuke-eval-runtime python3 experiments/pannuke_d5/calibrate_pixel_wcsp.py evaluate --selection experiments/pannuke_d5/pixel_wcsp_extended_inputs/selection.csv --prepared experiments/pannuke_d5/pixel_wcsp_extended_inputs --maps experiments/pannuke_d5/pixel_wcsp_extended_maps --frozen-config experiments/pannuke_d5/pixel_wcsp_strength_calibration/frozen_config.json --output /tmp/pannuke_extended_evaluation
python3 experiments/pannuke_d5/summarize_pixel_evaluation.py --run /tmp/pannuke_extended_evaluation --prepared experiments/pannuke_d5/pixel_wcsp_extended_inputs --selection experiments/pannuke_d5/pixel_wcsp_extended_inputs/selection.csv --frozen-config experiments/pannuke_d5/pixel_wcsp_strength_calibration/frozen_config.json
```
