# Deliverable 6 — What Changed This Week

Covers only work added since Deliverable 5 was finished on Monday, October 26: all the commits from November 1–2.

## Summary

| Area | What we tried | Result |
|---|---|---|
| PanNuke | Export HoVer-Net's real pixel probabilities and use them as unary costs | Now possible, but raw probabilities alone score below HoVer-Net |
| PanNuke | Two-stage WCSP: find nuclei, then type each nucleus | Dev mIoU 0.4749 → 0.4325: worse, dropped |
| PanNuke | One-stage WCSP on all pixels, tested on 100 unused patches | mIoU +0.0003, CI includes 0: no reliable change |
| PanNuke | Learned b table, tested on 20 patches | mIoU −0.0017, CI includes 0: no reliable change |
| PanNuke | β sweep for the learned table on dev | β = 0.15 is best, +0.0016 mIoU; not yet tested on fresh patches |
| BCSS | Retrain the U-Net with cross-entropy | Dev mIoU 0.6362 → 0.6679 |
| BCSS | Full-resolution head + backbone | Dev mIoU 0.6797, **the best result this week** |
| BCSS | Add a pairwise loss during training | No gain over cross-entropy |
| BCSS | Add distant pixel pairs to the WCSP | No gain, and slower |
| BCSS | toulbar2 with a 2 s per-block limit | No gain over the U-Net; ICM → toulbar2 ties ICM |

**Main takeaway:** improving the classifier gained far more than improving the WCSP. Training gained about +0.04 dev mIoU, while the WCSP adds about +0.001 to +0.003.

## PanNuke

### What changed

Last Monday's flat unary couldn't use HoVer-Net's confidence, so this week started by exporting the real pixel probabilities.

| | Deliverable 5 | This week |
|---|---|---|
| Unary | Flat penalty for changing a label | −log p, so it counts certainty |
| Variables | Nucleus pixels only | All 26,896 pixels, including background |
| Labels | 5 nucleus classes | Those 5 + background |
| b table | Handwritten | Handwritten, then learned |

### Step 1: find a formulation (15 dev patches)

Saved HoVer-Net scores mean mIoU **0.4749**. The handwritten table is used here, with background added.

| Method | mIoU | Δ vs. HoVer-Net |
|---|---:|---:|
| Two-stage WCSP: find nuclei, then type each nucleus | 0.4325 | −0.0424 |
| Pixel argmax of the probabilities, no WCSP | 0.4438 | −0.0311 |
| Six-class type head + WCSP, starting from its own argmax | 0.4410 | −0.0339 |
| Six-class type head + WCSP, starting from HoVer-Net, neighbor strength 2 | 0.4624 | −0.0125 |
| One-stage WCSP, nucleus × type probabilities, starting from HoVer-Net | **0.4759** | **+0.0010** |

- **Two-stage:** a nucleus removed in the first stage can't come back, and it scored worst, so it was dropped.
- **HoVer-Net's labels are hard to beat:** they include instance post-processing, and raw pixel probabilities don't.
- **Neighbor strength 2 is the highest value tried:** it gave the best WCSP score of the strengths tested on the six-class type head, but still fell below HoVer-Net.

### Step 2: test it on unused patches (handwritten table, neighbor strength 2)

Settings were frozen before these runs, using the six-class type-head WCSP starting from HoVer-Net.

| Test set | HoVer-Net mIoU | + WCSP mIoU | Δ [95% CI] | Nucleus IoU |
|---|---:|---:|---|---:|
| 20 patches | 0.5555 | 0.5599 | +0.0044 [−0.0089, +0.0181] | 0.7709 → 0.7649 |
| 100 patches, 18 tissues | 0.5951 | 0.5954 | +0.0003 [−0.0078, +0.0082] | 0.7433 → 0.7392 |

- **The larger test removes most of the early gain:** on 100 patches, 44 improve and 55 worsen.
- **Pixel changes nearly cancel out:** 17,953 pixels fixed and 16,935 broken.
- **Nucleus detection gets slightly worse** on both sets.

### Step 3: learn the b table

Next the team replaced the handwritten table with a learned one: 80% from 200 training patches, 20% biological prior. The baseline here is the raw argmax of the 6-class output, without HoVer-Net's post-processing.

**Learned b table** (upper triangle; same label costs 0):

| | Background | Neoplastic | Inflammatory | Connective | Dead | Non-neoplastic epithelial |
|---|---:|---:|---:|---:|---:|---:|
| **Background** | 0 | 0.42 | 0.36 | 0.36 | 0.22 | 0.41 |
| **Neoplastic** | | 0 | 1.04 | 0.84 | 1.29 | 2.31 |
| **Inflammatory** | | | 0 | 0.68 | 1.13 | 1.93 |
| **Connective** | | | | 0 | 1.36 | 1.06 |
| **Dead** | | | | | 0 | 1.60 |
| **Non-neoplastic epithelial** | | | | | | 0 |

**Test result (20 held-out patches, β = 0.30).** This uses pooled mIoU and a different baseline, so it can't be compared with Step 2.

| Metric | Raw argmax | + WCSP |
|---|---:|---:|
| Biological mIoU | 0.3769 | 0.3752 |
| Six-class mIoU | 0.4731 | 0.4718 |
| Pixel accuracy | 0.9334 | 0.9336 |

- **mIoU change:** −0.0017, 95% CI [−0.0050, +0.0080], so no reliable improvement.
- **Fixes and breaks:** 801 pixels fixed and 687 broken, a net gain of 114. That raises pixel accuracy, but inflammatory, connective and dead IoU fall, and mIoU weights each class equally.

### Step 4: sweep β for the learned table (20 separate dev patches)

β = 0.30 was carried over from the earlier handwritten-table sweep, so the team re-swept 15 values for the learned table.

| | Raw argmax | β = 0.15 | β = 0.30 |
|---|---:|---:|---:|
| Biological mIoU | 0.4258 | **0.4274** | 0.4270 |
| Pixel accuracy | 0.9056 | **0.9059** | 0.9058 |
| Fixes / breaks | — | 574 / 421 | 813 / 744 |

![β sweep](../experiments/pannuke_wcsp/pannuke_learned_table_beta_dev/beta_sweep.png)

β = 0.15 is a development pick. It still needs a fresh test set, because the 20 test patches above have already been used.

## BCSS

### New data preparation

Tiles were regenerated from the original images: 7,768 fit, 1,012 dev and 4,028 val, with the same 106/45 hospital-held-out split of tissue regions. **These scores can't be compared directly with Deliverable 5**, which used a different tiling. All of them are dev results; none of the retrained models has a val result yet.

### Training the U-Net (all 1,012 dev tiles, central 256 × 256)

| Model | mIoU | Δ vs. cross-entropy |
|---|---:|---:|
| Pretrained U-Net, no training | 0.6362 | −0.0317 |
| Cross-entropy | 0.6679 | — |
| Full-resolution head + backbone, cross-entropy | **0.6797** | **+0.0117** |
| Cross-entropy + 0.01 × pairwise loss | 0.6597 | −0.0082 |
| Cross-entropy + 0.1 × pairwise loss | 0.6582 | −0.0097 |
| Cross-entropy + 1 × pairwise loss | 0.6654 | −0.0026 |

- All arms ran 10 epochs on one seed, with the checkpoint chosen on dev.
- The full-resolution arm changes architecture and training field together, so its gain can't be credited to resolution alone.

### WCSP on top of the trained models (handwritten b, λ = 8)

These use a 200-tile dev selection. Only rows within one table can be compared.

| Scored on | Model | U-Net mIoU | + WCSP mIoU | s/tile |
|---|---|---:|---:|---:|
| Central field, 181 tiles | Pretrained | 0.6017 | 0.6042 | 26.6 |
| Central field, 181 tiles | Cross-entropy | 0.6441 | 0.6452 | 17.8 |
| Central field, 181 tiles | Cross-entropy + 0.01 × pairwise loss | 0.6245 | 0.6243 | 18.0 |
| Whole tile, 185 tiles | Pretrained | 0.6223 | 0.6255 | 153.2 |
| Whole tile, 185 tiles | Cross-entropy | 0.6182 | 0.6196 | 240.7 |
| Whole tile, 185 tiles | Full-resolution head | **0.6524** | **0.6545** | 109.2 |

The WCSP still adds only about +0.001 to +0.003, whichever model it starts from. The one exception is the pairwise-loss model, where it adds nothing.

### Other WCSP tests (pretrained U-Net, central field, 181 dev tiles, mIoU 0.6017 before WCSP)

| Test | + WCSP mIoU | s/tile |
|---|---:|---:|
| ICM | 0.6042 | 24.7 |
| toulbar2, 2 s per block | 0.6017 | 9.4 |
| ICM → toulbar2, 2 s per block | 0.6042 | 37.1 |
| ICM on the whole grid, no distant pairs | 0.6042 | 22.0 |
| Whole-grid ICM + distant pairs (beyond 8 or 32 px) | 0.6041–0.6042 | 41.3–66.6 |

- **Distant pairs:** extra pixel pairs beyond 8 or 32 px, sampled at random, add runtime with no accuracy gain at the tested sampling rates.
- **toulbar2:** at 2 s per block it gives the same mIoU as the U-Net alone.

## Correction to Deliverable 5

The image term f was only scored on dev when Deliverable 5 was written, and it hurt there. Its val run has since finished: on the same 2,124 val tiles, f ties no f (mIoU 0.6507 and +0.0019 for both).

## Not finished

- **BCSS:**
  - Pairwise-loss weights 0.1 and 1 had no WCSP result at the last results refresh (Nov 2).
  - toulbar2 with no time limit, with and without ICM first, was also incomplete at that refresh.
- **PanNuke:** β = 0.15 needs fresh test patches.

## Next

1. Run the full-resolution model on the held-out val set, with the settings frozen first.
2. Test PanNuke β = 0.15 on patches that haven't been used.
3. Put effort into the classifier, since that's where this week's gain came from. WCSP changes have stayed small on both datasets.
