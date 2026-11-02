# Frozen neighbor-strength evaluation

Configuration and selection are frozen from the 15 development patches. The best actual WCSP candidate uses original six-channel TP probability unaries, saved HoVer-Net initialization and neighbor strength 2. The overall selected method remains unchanged HoVer-Net because no WCSP candidate beat it on development data.

| Method | Mean biological mIoU | Mean nucleus IoU |
|---|---:|---:|
| unchanged_hovernet | 0.555525 | 0.770946 |
| frozen_wcsp | 0.559888 | 0.764872 |

On these 20 new patches, the frozen actual WCSP candidate improves biological mIoU by 0.004363 (0.436 percentage points), improving 13 patches and worsening 7. Pixel accuracy rises from 0.927387 to 0.928646. It corrects 4,155 pixels and introduces 3,478 new errors. Binary nucleus IoU drops from 0.770946 to 0.764872. All 20 runs reach a single-pixel local minimum with nonincreasing energy.

The paired patch bootstrap 95% interval for mean mIoU change is [-0.008878, 0.018109] (10,000 resamples, seed 401). This is an exploratory uncertainty estimate; patch resampling does not account for possible patient correlation.

This is a small positive class-mIoU result on the new set, alongside a negative development result and worse binary segmentation. It does not demonstrate consistently equal-or-better performance, and the frozen selected method is not changed after inspecting evaluation results. The samples are from the same Fold 2 with patch-level exclusions, not verified patient-independent or external validation.

[Development settings and reproducible commands](../pixel_wcsp_strength_calibration/README.md) · [Per-patch metrics](per_patch.csv) · [Audit](audit.json)
