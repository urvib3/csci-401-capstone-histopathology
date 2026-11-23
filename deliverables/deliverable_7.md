# Deliverable 7 · November 23, 2026

**In short:** smoothing (WCSP) barely helps. Removing classes that shouldn't be in a tile (**presence**) helps more. The segmenter's real problems are **broken-up regions** and **whole regions given the wrong class**, so we are now training the U-Net to fix both directly.

---

## 1. What we've tried

mIoU on held-out data; change vs. the base model in brackets. Sources: PanNuke `organ_wcsp/breast/evaluation.json` (branch `organ-wcsp-breast`); BCSS val run summaries.

| Method | PanNuke (Breast test) | BCSS (val) |
|---|:---:|:---:|
| Base model | 0.4323 (HoVer-Net) | 0.6489 (U-Net) |
| Smoothing (WCSP, blocks, decay) | 0.4324 (+0.0001) | 0.6506 (+0.0018) |
| **Presence rule** | **0.4484 (+0.0161)** | **0.6550 (+0.0061)** |
| **Presence + cluster smoothing** | — | **0.6564 (+0.0075)** |

![Gains](figures_20261123/gains_both_datasets.png)

## 2. Why: our hypothesis

The model makes two kinds of mistakes:

| Mistake | Example | What fixes it |
|---|---|---|
| **Grouping** | One stroma region is predicted as 10 scattered pieces | Making regions coherent |
| **Labeling** | A whole region is called "other" when it's stroma | Knowing which classes belong in the tile |

Smoothing fixes neither. Presence fixes some labeling mistakes.

## 3. Tests that check this

On Ryan's full-resolution U-Net, 185 dev tiles, with no training.

| Test | Result |
|---|---|
| Share of wrong pixels that are **grouping** mistakes | **48%** |
| Share that are **labeling** mistakes | **52%** |
| Pieces per true region | **~10** (should be 1) |
| Presence + smoothing on this model | 0.6524 → **0.6610** (+0.0086) |
| If we knew the **true classes** per tile (upper bound) | — |
| If every true region were **coherent** (upper bound) | — |

![Mistakes per class](figures_20261123/prelim_error_split.png)

**Worst classes: "other" (36% wrong) and stroma (25%), and both are mostly labeling mistakes** (whole regions called the wrong class). Inflammatory and necrosis errors are mostly grouping.

## 4. What's running: fixing the U-Net itself

Same training as Ryan's model; only the loss changes. Score = raw U-Net, no post-processing.

| Model | Added loss | Fixes | Dev mIoU |
|---|---|---|:---:|
| A (Ryan) | none | — | 0.6679 |
| B | region loss: pixels in the same true region should agree | Grouping | — |
| C | presence loss: predict which classes are in the tile | Labeling | — |

**Success:** B or C beats A.
