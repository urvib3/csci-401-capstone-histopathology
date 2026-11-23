# Deliverable 7

*November 23, 2026 · PanNuke (cell nuclei) and BCSS (tissue regions)*

---

## Recap of results

mIoU, with the change vs. the base model in brackets. "—" = not run (yet).

| Method | PanNuke · Breast test | BCSS · dev | BCSS · val |
|---|:---:|:---:|:---:|
| Base model (HoVer-Net / U-Net) | 0.4323 | 0.7067 | 0.6489 |
| Pixel WCSP, handwritten table | — | 0.7097 (+0.0030) | 0.6507 (+0.0019) |
| Pixel WCSP, data-learned table | 0.4324 (+0.0001) | 0.7090 (+0.0023) | 0.6506 (+0.0018) |
| Hierarchical (multi-scale blocks) | 0.4323 (0.0000) | 0.7107 (+0.0040) | 0.6517 (+0.0028) |
| Hierarchical + decay | 0.4323 (0.0000) | 0.7108 (+0.0041) | *running* |
| Cluster (nucleus / SLIC superpixel) | 0.4344 (+0.0020) | 0.7163 (+0.0095) | *running* |
| **Presence** | — | **0.7229 (+0.0162)** | **0.6550 (+0.0061)** |
| **Presence + cluster** | **0.4484 (+0.0161)** | **0.7242 (+0.0175)** | **0.6564 (+0.0075)** |

![Results by method](figures_20261123/gains_both_datasets.png)

**Takeaway:** smoothing (pixel, hierarchical, decay) barely moves either dataset. Presence, which removes classes that shouldn't be in a tile, gives the only clear gain on both.

<sub>PanNuke: `experiments/pannuke_wcsp/organ_wcsp/breast/evaluation.json` (branch `organ-wcsp-breast`); there, "cluster" means one label per detected nucleus. BCSS: run summaries, λ = 8, SLIC superpixels for "cluster".</sub>

---

## Hypothesis

The segmenter makes two kinds of mistakes, and smoothing fixes neither.

| Mistake | Example | What would fix it |
|---|---|---|
| **Grouping** | One stroma region is predicted as ~10 scattered pieces | Coherent regions |
| **Labeling** | A whole region is called "other" when it's stroma | Knowing which classes belong in the tile |

---

## Preliminary tests

*Ryan's full-resolution U-Net, 185 dev tiles, no training.*

| Test | Result |
|---|---|
| Wrong pixels that are **grouping** mistakes | **48%** |
| Wrong pixels that are **labeling** mistakes | **52%** |
| Predicted pieces per true region | **~10** (should be 1) |
| Presence + smoothing on this model | 0.6524 → **0.6610** (+0.0086) |
| Upper bound: true classes given per tile | — |
| Upper bound: every true region coherent | — |

![Mistakes per class](figures_20261123/prelim_error_split.png)

**Worst classes: "other" (36% wrong) and stroma (25%), and both are mostly labeling mistakes.** Inflammatory and necrosis errors are mostly grouping.

---

## Current experiments: fixing the U-Net itself

Same training as Ryan's model; only the loss changes. Score = raw U-Net, no post-processing.

| Model | Added loss | Fixes | Dev mIoU |
|---|---|---|:---:|
| A (Ryan) | none | — | 0.6679 |
| B | Region loss: pixels in the same true region should agree | Grouping | — |
| C | Presence loss: predict which classes are in the tile | Labeling | — |

**Success = B or C beats A.**
