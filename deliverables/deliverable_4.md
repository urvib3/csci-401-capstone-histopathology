# Deliverable 4: Preliminary WCSP Integration for PanNuke and BCSS

## Objective

This deliverable advances our existing segmentation work toward WCSP refinement. We focus on connecting model predictions to a small optimization problem and checking that strong unary fidelity preserves the initial labels. We then examine the effect of weaker fidelity. Label changes alone do not demonstrate improved segmentation accuracy.

From our previous findings, PanNuke inference and prediction inspection are available, and the BCSS U-Net baseline achieved approximately 0.649 mean IoU and 0.779 mean Dice. The contributions below build on those outputs through new integration work and numerical checks.


## Task 1 — PanNuke: Prepare a Real Prediction Crop

We selected a real 3×3 crop from the existing `hovernet_fast-pannuke` predictions for PanNuke Fold 2. The crop was taken from `image_00748` at HoVer-Net output coordinates `x = 5–7, y = 17–19`. Because the fast HoVer-Net output represents the central 164×164 region of the original 256×256 patch, a 46-pixel offset was applied in both directions, giving original-image coordinates `x = 51–53, y = 63–65`.

The corresponding instance-ID crop contained three detected nuclei: instances 5, 8, and 10. Using the saved prediction metadata, these instances were mapped to Non-neoplastic epithelial, Connective, and Inflammatory, respectively. Each pixel belonging to a detected nucleus was assigned that nucleus's predicted class. Instance-map value `0` was treated as inactive background and excluded from the WCSP rather than being interpreted as healthy tissue. The resulting crop contains eight active pixels, three predicted biological classes, and one inactive background pixel, making it suitable as a small real-model input for the preliminary WCSP experiment.

### Predicted 3×3 Label Grid

|       | x=5 | x=6 | x=7 |
|------:|:---:|:---:|:---:|
| y=17  | 5 | 3 | 3 |
| y=18  | 0 | 3 | 3 |
| y=19  | 2 | 2 | 3 |

**Label key:**  
- `0` — Background / inactive  
- `2` — Inflammatory  
- `3` — Connective  
- `5` — Non-neoplastic epithelial  

The full five-class WCSP domain remains Neoplastic, Inflammatory, Connective, Dead, and Non-neoplastic epithelial; classes 1 and 4 simply do not appear in this selected crop.

## Task 2 — PanNuke: Construct and run the tiny WCSP

We used Task 1's real crop from `image_00748` without changing its initial labels. The eight nonzero pixels became variables, each with domain `{1, 2, 3, 4, 5}`. Background remained inactive. The five neighborhood types produced 28 unique undirected edges: 9, 6, 5, 6, and 2 edges at distances 1, √2, 2, √5, and 2√2, respectively. Interactions outside the crop were excluded.

Unary cost was `lambda` for changing a pixel's initial label and zero otherwise. Pairwise costs used the supplied base matrix divided by pixel distance. The unspecified attribute term was set to zero and the distance exponent to one for this preliminary check. Costs were evaluated without scaling or rounding.

We enumerated all **390,625 assignments** per setting, obtaining exact global optima for this crop. Unary-only used `lambda = 1` with pairwise terms disabled. Strong fidelity used `lambda = 193`, exceeding the conservative maximum-pairwise-cost bound of **191.330290**; any changed pixel therefore costs more than all possible pairwise savings. Weaker fidelity used `lambda = 1` with pairwise terms enabled. No time limit or additional optimization rules were applied. Implementation, returned label arrays, and numerical outputs are stored in [`experiments/pannuke_d4/pannuke_task2/`](../experiments/pannuke_d4/pannuke_task2/).

## Task 3 — PanNuke: Verify and interpret the results

An independent verifier loaded Task 1's initial labels, rebuilt edges using neighbor offsets, and recomputed unary and pairwise costs. It separately enumerated all assignments to verify optimality and count ties. All checks passed within absolute tolerance `1e-9`; each setting had exactly one optimum. Background stayed zero.

- **Unary-only:** initial and returned objectives were **0**; no pixels changed. Runtime: **0.133 s**.
- **Strong fidelity:** initial and returned objectives were **22.804398**, comprising zero unary cost and 22.804398 pairwise cost; no pixels changed. Runtime: **0.183 s**.
- **Weaker fidelity:** objective decreased from **22.804398 to 3.000000**, comprising unary cost 3 and pairwise cost 0. Three pixels changed. Runtime: **0.179 s**.

Each runtime includes assignment generation, cost evaluation, and optimum selection. Initial and returned objectives are compared within the same setting.

```text
Initial / unary-only / strong      Weaker fidelity
5 3 3                             3 3 3
0 3 3                             0 3 3
2 2 3                             3 3 3
```

Strong fidelity reproduced the model labels. Weaker fidelity changed the Non-neoplastic epithelial pixel and both Inflammatory pixels to Connective, eliminating pairwise penalties. This demonstrates smoothing under the specified costs, not improved biological accuracy. No ground-truth evaluation was performed. The verifier and its recorded checks are stored in [`experiments/pannuke_d4/pannuke_task3/`](../experiments/pannuke_d4/pannuke_task3/).

## Task 4 — BCSS: Prepare input and resolve label handling

**Input.** The pretrained TIAToolbox U-Net (`fcn_resnet50_unet-bcss`), used as is (no training, no calibration), was run on BCSS_512 tiles on CPU (~0.4 s/tile on 4 cores). For each 512 × 512 tile it outputs five class probabilities for every pixel of the central 256 × 256, the same area the 0.649-mIoU baseline was scored on. These full softmax maps are saved, so unary costs are real probabilities, not hard labels.

**Label handling.** BCSS classes are reordered into the handout's domain: tumor → Neoplastic, inflammatory → Inflammatory, stroma → Connective, necrosis → Dead, other → Non-neoplastic. The last mapping is **provisional**: BCSS "other" mixes normal ducts with fat, vessels and blood, so it is not truly healthy tissue. Outside-ROI and excluded pixels are never scored. Unlike the PanNuke crop, every pixel has a label, so there is no inactive background.

**Code.** `wcsp/` (see `wcsp/README.md`) implements the full-size version; the tiny enumeration approach does not scale beyond ~9 pixels (5^n labelings).

## Task 5 — BCSS: Run the WCSP experiment

Each tile becomes a WCSP with 65,536 variables (one per pixel) and 778,770 pairwise cost functions (all five neighbor types). Cost = −log p from the U-Net + λ × (handout table ÷ distance), with η = 1 and the attribute term f = 0. It is solved with the handout's alternating-block scheme (four 100 × 100 corners, then the 56-px cross, repeated), using a greedy pixel-by-pixel solver (ICM) inside each block; a change is kept only if it lowers the total cost.

Scored against pathologist labels on **500 val tiles** (evenly spread over all 45 val ROIs; hospitals unseen in training):

| Method | Pixel accuracy | mIoU | Tumor | Stroma | Inflamm. | Necrosis | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net alone | 0.8397 | 0.6414 | 0.8088 | 0.7274 | 0.6017 | 0.6401 | 0.4291 |
| U-Net + WCSP (λ = 4) | 0.8403 | 0.6428 | 0.8090 | 0.7277 | 0.6031 | 0.6405 | 0.4336 |

Class columns are IoU. mIoU gain: **+0.0013**, 95% interval [+0.0003, +0.0023] from resampling the 45 val ROIs (gain ≤ 0 in 0.65% of resamples). The WCSP changed 0.9% of pixels and took ~8 s/tile on one CPU core.

**Reading the result.** The WCSP is consistently, but only slightly, better than the U-Net: the gain is small yet its interval excludes zero. Every class improves a little, most for inflammatory and "other". The U-Net output is already smooth, so the rules only move pixels on region boundaries; they cannot fix a whole region the U-Net got wrong. (An earlier 10-tile check gave the same picture: 0.798 → 0.800 accuracy.)

## Task 6 — BCSS: Verify results and specify the next implementation step

**Verification.** The cost function is checked against a slow pair-by-pair calculation of the handout's formula, toulbar2 against brute-force enumeration on small grids, and every solver returns exactly the U-Net's labels when fidelity is very large (44 automated tests, `tests/wcsp/`).

**Block strategy, as implemented.** Fixed pixels are not variables: their pairwise costs with free neighbors are added to those neighbors' unary costs, and fixed–fixed costs become a constant, so a block's cost plus the constant always equals the full image's cost. The four corners are independent because the cross (56 px) is wider than the longest interaction (2 px). A block's new labels are kept only if they lower the total cost, so the cost never increases; rounds stop when a full round brings no improvement (or after 4 rounds).

**Open issue.** The gain is small because boundary-only refinement is all the rules can do on an already-smooth U-Net. Next: tune λ on the dev split, try the table derived from BCSS statistics, and run all 2,768 val tiles.

## Next steps

During the first week, we will resolve dataset-specific label and background handling, retain usable prediction probabilities where available, and verify real-input fidelity checks on both datasets. During the second week, we will evaluate refinement against aligned ground truth and test alternating-block optimization on larger patches. Parameter selection will use development data, with held-out data reserved for evaluation.

<!-- Submission format: replace each task's instructions with its findings. Keep the complete report at or below 1,600 words, with no more than four small label-grid figures in total. Use readable 11–12 pt text, normal margins, and check the exported document is at most five pages. If it exceeds five pages, shorten prose or figures rather than shrinking text. Mark incomplete work explicitly; do not leave the task instructions in the submitted version. -->
