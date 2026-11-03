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
| BCSS | toulbar2 without a search timer, alone or after ICM | Both jobs hit the 8-hour allocation limit; no complete accuracy result |

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
| Central field, 181 tiles | Cross-entropy + 0.1 × pairwise loss | 0.6241 | 0.6244 | 15.5 |
| Central field, 181 tiles | Cross-entropy + 1 × pairwise loss | 0.6293 | 0.6291 | 13.9 |
| Whole tile, 185 tiles | Pretrained | 0.6223 | 0.6255 | 153.2 |
| Whole tile, 185 tiles | Cross-entropy | 0.6182 | 0.6196 | 240.7 |
| Whole tile, 185 tiles | Full-resolution head | **0.6524** | **0.6545** | 109.2 |

For the pretrained and cross-entropy models, WCSP adds about +0.001 to +0.003 mIoU. All three pairwise-loss evaluations are now complete: WCSP changes mIoU by −0.0002, +0.0003 and −0.0002 at weights 0.01, 0.1 and 1, respectively. None beats the cross-entropy control on these same 181 dev tiles.

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
- **Without a search timer:** both toulbar2 jobs (alone and after ICM) hit their 8-hour SLURM limit on November 2. Streaming traces record 14 and 13 finished tiles, respectively, but neither run produced a complete shard (0/32 for each). Aggregate mIoU and mean runtime remain unavailable; these jobs are no longer running. The timeout messages are in `data/logs/bcss-toulbar-counts_6715091_{0,1}.out` and `data/logs/bcss-hybrid-counts_6715092_{0,1}.out`.

## Correction to Deliverable 5

The image term f was only scored on dev when Deliverable 5 was written, and it hurt there. Its val run has since finished: on the same 2,124 val tiles, f ties no f (mIoU 0.6507 and +0.0019 for both).

## Not finished

- **BCSS:**
  - toulbar2 without a search timer, with and without ICM first, timed out before producing complete evaluation shards.
  - The retrained models still need a held-out val evaluation with frozen settings.
- **PanNuke:** β = 0.15 needs fresh test patches.

## Next

1. Run the full-resolution model on the held-out val set, with the settings frozen first.
2. Test PanNuke β = 0.15 on patches that haven't been used.
3. Put effort into the classifier, since that's where this week's gain came from. WCSP changes have stayed small on both datasets.

<!-- BEGIN BCSS FOLLOWUP RESULTS -->
## BCSS follow-up experiments (development results)

Automatically refreshed from experiment artifacts at **2026-11-03 00:30 UTC**. Incomplete runs are explicitly marked; their partial metrics are not treated as final results.

**Protocol.** The follow-up uses the same underlying 151 BCSS ROIs and the same 106/45 hospital-held-out ROI split. Tiles were regenerated directly from the upstream images at requested 0.25 MPP: 7,768 fit, 1,012 dev, and 4,028 val tiles. Preparation version 2 preserves both axes of 16-bit grayscale masks; codes 0 and 7 are ignored. The documented earlier preparation has different tile counts, so the historical scores in [Deliverable 5](deliverable_5.md) are not a paired comparison to these follow-ups. These new results are **development results**, not a new held-out-validation result.

**Training.** Cross-entropy (CE) is the supervised pixel-classification loss. The control fine-tunes the original half-resolution U-Net with CE. The full-resolution arm trains the existing full-resolution RGB encoder/decoder refinement head **and** the pretrained backbone on the entire 512×512 tile; this is one forward pass, not shift-and-stitch. Pairwise-loss arms retain the original architecture and central field, adding a differentiable expected compatibility-table cost to CE. All arms use 10 epochs, batch 4, seed 0, and checkpoint selection on dev. The full-resolution checkpoint is selected by whole-tile dev mIoU; other checkpoints use central-crop dev mIoU.

| Training arm | Epochs logged / 10 | Best epoch | Central 256 mIoU | Whole 512 mIoU |
|---|---:|---:|---:|---:|
| Pretrained U-Net reference | 0 | 0 | 0.6362 | — |
| Original architecture, cross-entropy | 10/10 | 8 | 0.6679 | — |
| Full-resolution head + backbone, cross-entropy | 10/10 | 7 | 0.6797 | 0.6648 |
| Cross-entropy + 0.01 × pairwise | 10/10 | 8 | 0.6597 | — |
| Cross-entropy + 0.1 × pairwise | 10/10 | 10 | 0.6582 | — |
| Cross-entropy + 1 × pairwise | 10/10 | 10 | 0.6654 | — |

At the selected checkpoints, the full-resolution arm changes central-crop mIoU by +0.0117 relative to the CE control. Architecture and training field change together; this does not isolate resolution alone.
After all 10 epochs, the strongest pairwise-loss arm differs from the CE control by -0.0026 central-crop mIoU; none of the tested pairwise-loss weights improves on the CE control.

Training-table scores use all 1,012 dev tiles. Inference comparisons below use 200 evenly selected dev tiles, excluding tiles without labeled pixels in the evaluated field (181 central-field tiles; 185 whole-field tiles). Scores across those different populations must not be subtracted. Within each inference table, methods use the same scored field. WCSP uses raw probabilities, the handwritten table, λ=8, η=1, and local neighborhood normalization.

**Central-field model comparisons.**

| Method | Samples | U-Net mIoU | + WCSP mIoU | Mean solve s/tile | Status |
|---|---:|---:|---:|---:|---|
| Pretrained U-Net | 181 | 0.6017 | 0.6042 | 26.6 | Complete |
| Cross-entropy control | 181 | 0.6441 | 0.6452 | 17.8 | Complete |
| Pairwise weight 0.01 | 181 | 0.6245 | 0.6243 | 18.0 | Complete |
| Pairwise weight 0.1 | 181 | 0.6241 | 0.6244 | 15.5 | Complete |
| Pairwise weight 1 | 181 | 0.6293 | 0.6291 | 13.9 | Complete |

**Whole-field model comparisons.**

| Method | Samples | U-Net mIoU | + WCSP mIoU | Mean solve s/tile | Status |
|---|---:|---:|---:|---:|---|
| Pretrained U-Net, bilinear | 185 | 0.6223 | 0.6255 | 153.2 | Complete |
| Cross-entropy control, bilinear | 185 | 0.6182 | 0.6196 | 240.7 | Complete |
| Trained full-resolution head | 185 | 0.6524 | 0.6545 | 109.2 | Complete |

**Solver comparison on the same pretrained central-field probabilities.**

| Method | Samples | U-Net mIoU | + WCSP mIoU | Mean solve s/tile | Status |
|---|---:|---:|---:|---:|---|
| ICM, instrumented | 181 | 0.6017 | 0.6042 | 24.7 | Complete |
| Toulbar2, 2 s/block | 181 | 0.6017 | 0.6017 | 9.4 | Complete |
| ICM → Toulbar2, 2 s/block | 181 | 0.6017 | 0.6042 | 37.1 | Complete |
| Toulbar2, no search timer | — | — | — | — | Pending/incomplete: 0/32 shards |
| ICM → Toulbar2, no search timer | — | — | — | — | Pending/incomplete: 0/32 shards |

ICM permits at most 50 sweeps per block and stops early when no labels change. Block refinement permits four outer rounds and stops early on an energy plateau. The uncapped Toulbar2 arms have no internal search timer; the SLURM allocation still has an 8-hour wall limit. The earlier 2-second-capped searches are retained as separately labeled controls. A capped run that returns its starting labeling quickly is not evidence of faster optimization at comparable quality. Completing a conditioned block search does not establish a global optimum for the image.

- Toulbar2, no search timer: 14 tiles have finished according to streaming traces; 15 searches have returned with 1,668,242 nodes and 497,933 backtracks. Counters exclude searches still running; aggregate mIoU is withheld until all shards exist.
- ICM → Toulbar2, no search timer: 13 tiles have finished according to streaming traces; 11 searches have returned with 1,170,108 nodes and 257,527 backtracks. Counters exclude searches still running; aggregate mIoU is withheld until all shards exist.
Instrumented ICM used 9,320 sweeps over 1,692 block calls (5.51 sweeps/call); 3 calls reached the sweep cap. Outer stopping reasons: 151 energy plateaus and 30 round-limit stops. An outer energy plateau is not a proof of global optimality.

**Sampled distant relations.** Each unordered pair beyond Euclidean distance d is included independently with probability p. Extra edges retain the same compatibility table and distance decay, without diluting local weights or inverse-probability scaling. These runs use whole-grid ICM; the p=0 row is their matching solver control.

| Method | Samples | U-Net mIoU | + WCSP mIoU | Mean solve s/tile | Status |
|---|---:|---:|---:|---:|---|
| No distant edges (p=0) | 181 | 0.6017 | 0.6042 | 22.0 | Complete |
| d=8, p=0.000001 | 181 | 0.6017 | 0.6042 | 41.3 | Complete |
| d=8, p=0.00001 | 181 | 0.6017 | 0.6041 | 53.5 | Complete |
| d=32, p=0.000001 | 181 | 0.6017 | 0.6042 | 52.4 | Complete |
| d=32, p=0.00001 | 181 | 0.6017 | 0.6041 | 66.6 | Complete |

Observed distant-edge mIoU differences from the matching control range from -0.000039 to -0.000011. These settings show no observed accuracy gain; they add runtime. This is a dev observation, not a validation claim.

**Historical shift-and-stitch reference.** The recorded earlier 200-tile dev selection scored the central 256×256 field. Bilinear U-Net / shift-and-stitch mIoU were 0.7067 / 0.7071; with WCSP at λ=8, 0.7097 / 0.7139. Shift-and-stitch interleaves four shifted forward passes with no retraining. There is no fresh shift-and-stitch result on the current tile selection yet; do not rank these historical numbers against the new training table. Source: [BCSS benchmark notes](../benchmarks/bcss/README.md#addendum-changing-the-u-nets-output-resolution).

**Artifacts and interpretation.** Experiment definitions and reproduction commands are in [the BCSS runbook](../experiments/bcss_wcsp/TONIGHT.md). Current logs, predictions, per-tile CSVs, streaming iteration traces, and the machine-readable report snapshot are under `data/outputs/tonight/` and `data/tonight/` on the cluster. Runtime is observed per-tile wall time under concurrent cluster load, not a controlled hardware microbenchmark. Dev checkpoint selection and parameter comparisons do not replace a separately frozen validation evaluation. Energy reduction, segmentation accuracy, and solver runtime are reported as distinct outcomes.
<!-- END BCSS FOLLOWUP RESULTS -->
