# Deliverable 4: Preliminary WCSP Integration for PanNuke and BCSS

## Objective

This deliverable advances our existing segmentation work toward WCSP refinement. We focus on connecting model predictions to a small optimization problem and checking that strong unary fidelity preserves the initial labels. We then examine the effect of weaker fidelity. Label changes alone do not demonstrate improved segmentation accuracy.

From our previous findings, PanNuke inference and prediction inspection are available, and the BCSS U-Net baseline achieved approximately 0.649 mean IoU and 0.779 mean Dice. The contributions below build on those outputs through new integration work and numerical checks.


## Task 1 — PanNuke: Prepare a Real Prediction Crop [Aakanksha]

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

We enumerated all **390,625 assignments** per setting, obtaining exact global optima for this crop. Unary-only used `lambda = 1` with pairwise terms disabled. Strong fidelity used `lambda = 193`, exceeding the conservative maximum-pairwise-cost bound of **191.330290**; any changed pixel therefore costs more than all possible pairwise savings. Weaker fidelity used `lambda = 1` with pairwise terms enabled. No time limit or additional optimization rules were applied. Implementation, returned label arrays, and numerical outputs are stored in `pannuke_task2/`.

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

Strong fidelity reproduced the model labels. Weaker fidelity changed the Non-neoplastic epithelial pixel and both Inflammatory pixels to Connective, eliminating pairwise penalties. This demonstrates smoothing under the specified costs, not improved biological accuracy. No ground-truth evaluation was performed. The verifier and its recorded checks are stored in `pannuke_task3/`.

