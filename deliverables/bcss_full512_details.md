# BCSS full-resolution corrections · details

[← Back to Deliverable 7](deliverable_7.md)

<!-- BEGIN FULL512 BCSS -->
## BCSS full-resolution correction experiments

**Changing whole regions and using class presence helped most.** On development
data, region moves improved mIoU from **0.6524 to 0.6670**; presence reached
**0.6648**.

We reused one frozen U-Net on **185 full-resolution 512×512 dev tiles from
14 tissue regions**, without retraining. This is a different model and field
of view from the earlier central-256 results. We tested **17 optimization
settings** and **40 correction settings**. The November 23, 15:16 PST snapshot
contains 34 completed correction settings; the appearance sweep was incomplete.

### What we tried

Scores below use the same dev tiles. Each experimental row shows its best
completed setting; the appearance result is provisional.

| Method | What changes | Dev mIoU |
|---|---|---:|
| Raw U-Net | No correction | 0.6524 |
| Pixel WCSP | Smooth neighboring pixels | 0.6547 |
| Hierarchical WCSP | Link pixels to larger blocks | 0.6570 |
| Stronger spatial penalties | Tune pixel and block coupling strengths | 0.6577 |
| Higher temperature | Soften U-Net confidence (T=4) | 0.6582 |
| Remove block fidelity | Let blocks change class without their own U-Net penalty | 0.6583 |
| **Class presence** | Use training-derived class combinations to suppress unlikely classes | **0.6648** |
| **Connected-region moves** | Relabel a whole connected prediction region together | **0.6670** |
| Stain-aware boundaries | Encourage similar neighboring tissue colors to agree | 0.6570 |
| Separate appearance classifier | Add stain and texture evidence learned on training tiles | 0.6550 |

Region moves can escape solutions where changing any single pixel is too
expensive. Connected prediction regions worked better than SLIC proposals.
Presence uses a refitted class-set prior, extending the earlier presence pilot.
Both methods made larger gains than adjusting smoothing or running more
iterations. Higher temperature helped modestly; removing block fidelity was
less certain, with a confidence interval that includes zero. Stain-aware
boundaries did not consistently help.

These are **development-selected results**. The 95% intervals for gains over
matched controls were +0.0027 to +0.0221 for region moves and +0.0036 to +0.0127
for presence. They resample whole tissue regions and do not adjust for trying
multiple settings.

### What the corrections look like

![Translucent tissue-label overlays showing a region-move improvement and failure](figures_20261123/bcss_full512_region_examples.png)

Masks use 35% opacity so tissue remains visible. The final column marks edits
relative to pixel WCSP: **green = fixed, red = broken, orange = still wrong**;
unmarked and ignored pixels show the original image. These are the median
improvement and median failure among dev tiles with a score change, selected
after inspecting results. Region moves fixed 601,023 pixels but broke 404,681
across the cohort, relative to the raw U-Net: they can help substantially and
still introduce errors.

### What is confirmed on full validation?

The earlier evaluation accounts for **4,028 validation tiles from 45 tissue
regions**: 3,229 scored and 799 entirely ignored. Raw U-Net scored **0.6707**,
pixel WCSP **0.6724**, and multiscale WCSP **0.6736** (+0.0029 over U-Net).
**The new presence and region-move candidates have not yet been tested on
this full validation set.** Their next step is evaluation with matched
controls after development selection is frozen.

[Correction results](../experiments/bcss_wcsp/results/full512_20261123/corrections_dev.json) ·
[Optimization sweep](../experiments/bcss_wcsp/results/full512_20261123/expanded_dev.json) ·
[Validation results](../experiments/bcss_wcsp/results/full512_20261123/validation.json) ·
[Protocol and reproducibility details](../experiments/bcss_wcsp/results/full512_20261123/README.md)
<!-- END FULL512 BCSS -->
