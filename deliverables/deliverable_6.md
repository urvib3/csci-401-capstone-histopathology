# Deliverable 6 — What Changed This Week

Covers the commits from November 1–2, plus one BCSS run left over from Deliverable 5. Everything else is in [Deliverable 5](deliverable_5.md).

## Summary

| Area | What we tried | Result |
|---|---|---|
| PanNuke | WCSP on HoVer-Net's real probabilities, with a learned b table | Test mIoU −0.0017, CI includes 0: no reliable change |
| PanNuke | Sweep β for that table on dev | β = 0.15 is best, +0.0016 mIoU on dev; not yet tested on fresh patches |
| BCSS | Retrain the U-Net with cross-entropy | Dev mIoU 0.6362 → 0.6679 |
| BCSS | Full-resolution head + backbone | Dev mIoU 0.6797, **the best result this week** |
| BCSS | Add a pairwise loss during training | No gain over cross-entropy |
| BCSS | Add distant pixel pairs to the WCSP | No gain, and slower |
| BCSS | toulbar2 with a 2 s per-block limit | No gain over the U-Net; ICM → toulbar2 ties ICM |
| BCSS | Image term f on all val tiles | Ties no f |

**Main takeaway:** improving the classifier gained far more than improving the WCSP. Training gained about +0.04 dev mIoU, while the WCSP adds about +0.001 to +0.003.

## PanNuke

### What changed

| | Deliverable 5 | This week |
|---|---|---|
| Starting labels | HoVer-Net nucleus labels | Raw argmax of HoVer-Net's 6-class pixel output |
| Variables | Nucleus pixels only | All 26,896 pixels, including background |
| Unary | Flat penalty for changing a label | −log p, so it counts certainty |
| b table | Handwritten | 80% learned from 200 training patches, 20% biological prior |
| Trade-off | λ on the unary | β on the pairwise |

**Learned b table** (upper triangle; same label costs 0):

| | Background | Neoplastic | Inflammatory | Connective | Dead | Non-neoplastic epithelial |
|---|---:|---:|---:|---:|---:|---:|
| **Background** | 0 | 0.42 | 0.36 | 0.36 | 0.22 | 0.41 |
| **Neoplastic** | | 0 | 1.04 | 0.84 | 1.29 | 2.31 |
| **Inflammatory** | | | 0 | 0.68 | 1.13 | 1.93 |
| **Connective** | | | | 0 | 1.36 | 1.06 |
| **Dead** | | | | | 0 | 1.60 |
| **Non-neoplastic epithelial** | | | | | | 0 |

### Test result (20 held-out patches, β = 0.30)

| Metric | Raw argmax | + WCSP |
|---|---:|---:|
| Biological mIoU | 0.3769 | 0.3752 |
| Six-class mIoU | 0.4731 | 0.4718 |
| Pixel accuracy | 0.9334 | 0.9336 |

- **mIoU change:** −0.0017, 95% CI [−0.0050, +0.0080], so no reliable improvement.
- **Fixes and breaks:** 801 pixels fixed and 687 broken, a net gain of 114. That raises pixel accuracy, but inflammatory, connective and dead IoU fall, and mIoU weights each class equally.
- **Compared with last week:** the WCSP now makes real changes, including about as many fixes as breaks. Last week's flat-unary runs on the 15 dev patches broke far more than they fixed (at λ = 1, 1 fix and 28 breaks). The two runs use different patches and baselines, so this isn't a paired comparison.

### β sweep (20 separate dev patches)

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

### Image term f on val (leftover from Deliverable 5)

Scored on all 2,124 val tiles with the Deliverable 5 tiling, handwritten b, λ = 8.

| Method | Pixel acc | mIoU | Δ mIoU vs. U-Net [95% CI] |
|---|---:|---:|---|
| U-Net | 0.8414 | 0.6489 | — |
| + WCSP, no f | 0.8419 | 0.6507 | +0.0019 [+0.0010, +0.0027] |
| + WCSP, f (β = 1) | 0.8419 | 0.6507 | +0.0019 [+0.0009, +0.0027] |

This corrects Deliverable 5, where the f row is dev-only and says f hurts. On val, f ties no f, at about twice the runtime (25 vs. 11 s per tile).

## Not finished

- **BCSS:**
  - Pairwise-loss weights 0.1 and 1 had no WCSP result at the last results refresh (Nov 2).
  - toulbar2 with no time limit, with and without ICM first, was also incomplete at that refresh.
  - Handwritten b at λ = 16 on val and ICM → toulbar2 on small tiles both stopped partway when some jobs failed, so neither is reported.
- **PanNuke:** β = 0.15 needs fresh test patches.

## Next

1. Run the full-resolution model on the held-out val set, with the settings frozen first.
2. Test PanNuke β = 0.15 on patches that haven't been used.
3. Put effort into the classifier, since that's where this week's gain came from. WCSP changes have stayed small on both datasets.
