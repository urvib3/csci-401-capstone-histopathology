# Deliverable 7

**Web version:** https://urvib3.github.io/csci-401-capstone-histopathology/deliverable7/

*November 23, 2026 · PanNuke (cell nuclei) and BCSS (tissue regions)*

---

## Recap of results

mIoU, with the change vs. the base model in brackets. "—" = not run (yet).

| Method | PanNuke · Breast test | BCSS · dev | BCSS · val |
|---|:---:|:---:|:---:|
| Base | 0.4323 | 0.7067 | 0.6489 |
| Pair (HW) | — | 0.7097 (+0.0030) | 0.6507 (+0.0019) |
| Pair (DD) | 0.4324 (+0.0001) | 0.7090 (+0.0023) | 0.6506 (+0.0018) |
| Hier | 0.4323 (0.0000) | 0.7107 (+0.0040) | 0.6517 (+0.0028) |
| Hier+Decay | 0.4323 (0.0000) | 0.7108 (+0.0041) | *running* |
| Cluster | 0.4344 (+0.0020) | 0.7163 (+0.0095) | *running* |
| **Presence** | — | **0.7229 (+0.0162)** | **0.6550 (+0.0061)** |
| **Pres+Clust** | **0.4484 (+0.0161)** | **0.7242 (+0.0175)** | **0.6564 (+0.0075)** |

**Base** HoVer-Net / U-Net · **Pair (HW / DD)** pixel WCSP, handwritten / data-derived table · **Hier** multi-scale blocks · **Hier+Decay** with level decay · **Cluster** one label per nucleus (PanNuke) or SLIC superpixel (BCSS) · **Presence** drop classes that shouldn't be in the tile · **Pres+Clust** both

![Results by method](figures_20261123/gains_both_datasets.png)

**Takeaway:** smoothing (pixel, hierarchical, decay) barely moves either dataset. Presence, which removes classes that shouldn't be in a tile, gives the only clear gain on both.

<sub>PanNuke: `experiments/pannuke_wcsp/organ_wcsp/breast/evaluation.json` (branch `organ-wcsp-breast`); there, "cluster" means one label per detected nucleus. BCSS: run summaries, λ = 8, SLIC superpixels for "cluster".</sub>

---

## Problems

**The U-Net has limitations, and WCSP right now is only a smoother, so it can't fix them.** Smoothing cleans up noisy borders; the real errors are bigger.

| Problem | Dataset | What goes wrong | What could fix it |
|---|---|---|---|
| **Segmentation** | BCSS | One true region is broken into many pieces | Train the U-Net to keep regions whole |
| **Class identification** | BCSS | A whole region gets the wrong class, or a class that isn't in the tile | Strong data-derived rules (e.g. presence), not light smoothing |
| **Mixed tissue types** | PanNuke | Patches from all organs share one model and one prior, so no organ is fit well | Run per tissue type: fit and tune on one organ at a time |

![BCSS examples of the two problems](figures_20261123/issue_examples.png)


---

## BCSS preliminary tests

**Goal:** find out whether **segmentation** or **class identification** is the bigger problem for the U-Net, so we know what to fix first.

*Full-resolution U-Net · 185 BCSS dev tiles · no training*

### The two errors

- **Segmentation error:** the U-Net knows what a region is, but gets its shape wrong: pieces break off, borders are off.
- **Class identification error:** the U-Net gets the shape roughly right, but calls the region the wrong tissue.

### Method

Take each true region (one connected piece of one class in the ground truth) and check the U-Net's most common label inside it. **Right** → its wrong pixels are segmentation errors. **Wrong** → they are class identification errors.

![Method](figures_20261123/prelim_split_diagram.png)

### Test 1 · How much of each class is wrong, and why?

![Test 1](figures_20261123/prelim_error_split.png)

**Class identification is slightly bigger on average** (12.2% vs 9.4% of a class's pixels). It dominates for "other" and stroma, while segmentation dominates for inflammatory, necrosis and tumor.

### Test 2 · What would fixing each one be worth?

- **Perfect class identification:** keep the U-Net's regions; give each the class most of its true pixels have.
- **Perfect segmentation:** keep the true regions; give each the class the U-Net predicts for most of it.

![Test 2](figures_20261123/prelim_ladder.png)

**Both matter about equally:** perfect class identification +0.1186, perfect segmentation +0.1028. Presence (+0.0086) and smoothing (+0.0023) recover only a small fraction.

---

## Current experiments: fixing the U-Net itself

Same training recipe as the cross-entropy baseline; only the loss changes. Score = raw U-Net, no post-processing.

| Model | Added loss | Fixes | Dev mIoU |
|---|---|---|:---:|
| A (baseline) | none | — | 0.6679 |
| B | Region loss: pixels in the same true region should agree | Segmentation | — |
| C | Presence loss: predict which classes are in the tile | Class identification | — |

**Success = B or C beats A.**
