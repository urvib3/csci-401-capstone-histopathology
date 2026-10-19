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

_Section budget: 180–220 words._

Use Task 1's crop to construct the pixel-level WCSP. Report the actual variable and unique edge counts, unary-cost choice, pairwise settings, and solver/method. State implementation choices without repeating the handout's full matrix or neighborhood definitions.

Run unary-only, strong-fidelity, and weaker-fidelity settings. Select a strong weight with a documented sufficient bound, rather than assuming it preserves predictions. Report any cost scaling, solver limit, or unresolved issue. Keep supporting code and arrays on GitHub; describe the implemented behavior here.

If real-input or solver access is blocked, use a clearly labeled two-pixel synthetic example and identify the missing integration step.

<!-- Replace these task instructions with your completed contribution. -->

## Task 3 — PanNuke: Verify and interpret the results

_Section budget: 180–220 words and one compact before/after label grid._

Independently check the initial and returned assignments using Task 2's actual costs. Report whether strong fidelity preserved every initial label. For each setting, give the initial and returned objective values under that same setting, changed-pixel count, runtime, and solver status. These can be short bullets; a results table is optional.

Interpret weaker-fidelity changes and record any ties or numerical issues. Do not compare objective values across different fidelity weights as if they were the same objective. State what the experiment establishes and what still requires ground-truth evaluation.

<!-- Replace these task instructions with your completed contribution. -->

## Task 4 — BCSS: Prepare input and resolve label handling

_Section budget: 180–220 words and one small label grid._

Identify an actual per-pixel prediction artifact or obtain one tile's predictions using the existing model setup. Aggregate evaluation metrics cannot serve as prediction maps. Select a small crop and record the source, coordinates, class IDs, and excluded pixels. Verify alignment with the predicted central area.

Explain how BCSS tissue labels are represented in the WCSP. State any provisional compatibility mapping, especially the treatment of “other,” which must not automatically be equated with healthy tissue. Record whether full probabilities are available or a hard-label unary is needed. Share the crop and decisions with Tasks 5 and 6 early.

If input access or label compatibility remains unresolved, identify the blocker and distinguish any synthetic fallback from real BCSS integration.

<!-- Replace these task instructions with your completed contribution. -->

## Task 5 — BCSS: Run the tiny WCSP experiment

_Section budget: 200–250 words and one compact before/after label grid._

Reuse the shared implementation where possible and apply Task 4's explicit label decisions. Report variable and edge counts, unary/pairwise settings, the strong-fidelity bound, and the actual solver configuration.

Compare unary-only, strong-fidelity, and weaker-fidelity settings. Give initial and returned objective values under each setting, changed-pixel counts, runtime, and solver status. Identify provisional mappings or synthetic input explicitly. Explain what worked and what remains necessary before evaluating segmentation improvement.

Keep code and detailed arrays on GitHub. Summarize only the evidence needed to understand the result here.

<!-- Replace these task instructions with your completed contribution. -->

## Task 6 — BCSS: Verify results and specify the next implementation step

_Section budget: 180–220 words._

Independently check Task 5's objective calculations and strong-fidelity label preservation. Report the verification outcome and any discrepancy, tie, or timeout limitation.

Translate the handout's alternating-block strategy into concrete implementation decisions: fixed/free pixel handling, how interactions across block boundaries enter the subproblem, and a stopping rule. Focus on decisions needed by our implementation rather than restating the handout or redrawing its diagram. State which parts remain proposed.

Finish with the most important unresolved integration issue and the next test that would resolve it.

<!-- Replace these task instructions with your completed contribution. -->

## Next steps

During the first week, we will resolve dataset-specific label and background handling, retain usable prediction probabilities where available, and verify real-input fidelity checks on both datasets. During the second week, we will evaluate refinement against aligned ground truth and test alternating-block optimization on larger patches. Parameter selection will use development data, with held-out data reserved for evaluation.

<!-- Submission format: replace each task's instructions with its findings. Keep the complete report at or below 1,600 words, with no more than four small label-grid figures in total. Use readable 11–12 pt text, normal margins, and check the exported document is at most five pages. If it exceeds five pages, shorten prose or figures rather than shrinking text. Mark incomplete work explicitly; do not leave the task instructions in the submitted version. -->
