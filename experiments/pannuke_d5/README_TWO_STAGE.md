# PanNuke: two-stage WCSP

> Historical two-stage comparison. The current experiment is the [one-stage WCSP](README_FULLGRID.md).

`run_two_stage.py` separates nucleus/background segmentation from nucleus-type classification. Background means **non-nucleus**, not a background cell. Both stages use HoVer-Net predictions; this is not a standalone image classifier.

## Stage 1: nucleus or background

Every pixel in the 164 × 164 output region is a binary variable: 0 = background, 1 = nucleus. It can change in either direction.

The starting configuration is an **untuned baseline**, not learned biological costs:

| Original prediction | Choose background | Choose nucleus |
|---|---:|---:|
| Background | 0 | 2 |
| Nucleus | 2 | 0 |

These are the directed unary transition costs in `stage1_transitions`. The two change penalties can differ. For example, increasing background → nucleus makes expansion harder; increasing nucleus → background makes removal harder. Costs always refer to the original HoVer-Net prediction, not the previous solver update.

Each neighboring pixel pair pays **0.25 / distance** when one is background and one is nucleus, otherwise zero. The existing 24-neighbor geometry is used. This favors smoother masks; a change costing 2 is worthwhile only if the incident boundary savings exceed 2. The boundary weight 0.25 keeps smoothing weaker than the change penalty, but does not guarantee preservation of thin structures.

If `--nucleus-probabilities` supplies an aligned `(H,W)` array of actual P(nucleus), add **−log P(candidate)** to each candidate's unary cost. Set the transition matrix to zero to use probability evidence without an extra change penalty. Without this array, the model uses only hard-label fidelity and spatial smoothing; it cannot identify a completely missed isolated nucleus from image appearance.

## Between stages: recover nucleus instances

Keep surviving HoVer-Net instance markers, split disconnected surviving pieces, and expand markers into newly active pixels using four-connected breadth-first propagation. Distinct touching markers remain separate. Active components without any surviving marker become new candidate nuclei. This deterministic partition is not a learned boundary detector and can still split or merge objects incorrectly.

## Stage 2: type each nucleus

Each reconstructed nucleus is **one variable**, choosing classes 1–5: Neoplastic, Inflammatory, Connective, Dead, Non-neoplastic epithelial. Its whole mask gets the chosen type. Stage 2 cannot change the stage-1 mask.

- **Unary:** average the directed type-transition cost vectors of the original typed pixels retained in that nucleus. Rows are original classes, columns are candidate classes, in the order above. The current 5 × 5 matrix uses class-dependent directed penalties fitted on separate calibration patches (table below). Keeping the original type costs 0; every directed entry remains configurable. Averaging prevents larger nuclei from automatically having larger fidelity penalties.
- **Optional type evidence:** `--type-probabilities` accepts an aligned `(5,H,W)` array of conditional nucleus-type probabilities, summing to one per pixel. Add the mean **−log P(type)** across the nucleus. These must be actual model outputs, not probabilities fabricated from hard labels.
- **New nuclei:** without retained type evidence, require informative type probabilities. The pipeline fails explicitly instead of assigning an arbitrary class.
- **Pairwise:** connect distinct nuclei whose nearest pixel centers are at most **12 px** apart, even across background gaps. One edge per nucleus pair costs **strength × handwritten_table[type_A,type_B] / distance**. The starting strength is 1. Radius and strength are configurable, untuned choices.
- **Solver:** stage 1 uses the existing block/ICM solver. Stage 2 uses sequential nucleus-level ICM. Both are local searches; energies from the two different objectives are reported separately.

This stage-2 nucleus graph is a deliberate departure from the original pixel-level formulation.

## Class-dependent unary vectors (current default)

Rows = original HoVer-Net class; columns = candidate class. Each nucleus gets the corresponding row as its five-value lambda vector. For example, a Connective nucleus gets `[0.877, 0.794, 0, 1.705, 1.477]`. These are unary reclassification costs, not the adjacency table.

| Original → candidate | Neo | Inflam | Connective | Dead | Non-neo epithelial |
|---|---:|---:|---:|---:|---:|
| Neoplastic | 0 | 0.937 | 0.711 | 1.402 | 1.013 |
| Inflammatory | 1.155 | 0 | 0.710 | 1.681 | 1.681 |
| Connective | 0.877 | 0.794 | 0 | 1.705 | 1.477 |
| Dead | 0.631 | 0.154 | 0.235 | 0 | 1.123 |
| Non-neoplastic epithelial | 0.829 | 0.728 | 0.814 | 1.341 | 0 |

**Assignment rule:** count original-class → ground-truth-class evidence on **100 separate calibration patches (1,054 usable nuclei)**. Each nucleus contributes weight 1, distributed over its overlapping ground-truth nucleus types; predominantly false detections (less than 50% GT nucleus overlap) are excluded from type calibration. For original class `a` and candidate `b`, use:

`lambda[a,b] = log(1 + (C[a,a] + 1) / (C[a,b] + 1))` for `a != b`; diagonal = 0.

A correction observed rarely relative to keeping the original class gets a larger penalty. Add-one smoothing avoids zero-count infinities. Divide all off-diagonal values by one global mean so the average change cost stays **1**, matching the former flat baseline. Direction matters: Neoplastic → Connective is 0.711, whereas Connective → Neoplastic is 0.877.

Calibration excludes the 15 development patches, 20 frozen evaluation patches, and image_00748. Support by predicted class: Neo 503, Inflammatory 180, Connective 197, Dead 105, Non-neoplastic epithelial 69 nuclei. This is empirical patch-disjoint calibration, not a proven biological rule or a patient-independent split. Ground truth is used to fit this prior offline, never during a patch's optimization.

Costs and provenance: [configuration](two_stage_config.json), [calibration counts and selection](unary_calibration/fit.json). New run reports and `nuclei.csv` include each nucleus's actual unary vector. Earlier flat penalties remain reproducible with [the old configuration](two_stage_flat_costs_config.json).

## Run

Edit a copy of `two_stage_config.json` to change the costs. Output directories must be new or empty.

```bash
python experiments/pannuke_d5/run_two_stage.py \
  --source experiments/pannuke_d5/pannuke_preparation \
  --output /tmp/pannuke_two_stage \
  --config experiments/pannuke_d5/two_stage_config.json

python experiments/pannuke_d5/evaluate_fullgrid.py --run /tmp/pannuke_two_stage
```

Outputs include the binary mask, reconstructed instance map, initial and final type maps, per-nucleus class/area/centroid CSV, and the exact configuration. Optimization never reads ground truth; evaluation reports binary IoU/Dice separately from type mIoU and isolates stage-2 changes.

## Initial single-patch check (historical flat costs)

Saved trial: [image_00748](two_stage_trial_image00748/results.json), [evaluation](two_stage_trial_image00748/evaluation.json). This uses hard labels, without saved probability maps. The ground truth contains only Inflammatory nuclei.

| Measure | HoVer-Net | Two-stage WCSP |
|---|---:|---:|
| Binary nucleus IoU | 0.700519 | 0.700332 |
| Biological mIoU (only GT class here: Inflammatory) | 0.639899 | 0.692531 |
| Full-crop pixel accuracy | 0.799970 | 0.838452 |

Stage 1 adds 8 active pixels and removes none; 47 reconstructed nuclei remain. Stage 2 has 67 edges and retypes 8 nuclei. The final class map has 1,228 wrong → correct changes and 193 correct → wrong changes. Binary segmentation slightly worsens despite improved classification on this patch.

For context, the earlier untuned unified six-label trial scored 0.634316 biological mIoU on this patch. Its cost settings and variable representation differ, so that is **not a controlled comparison of two-stage versus unified optimization**. No settings here were selected through a development sweep.

## Development-set test (15 patches, historical flat costs)

Fixed costs were tested on the existing 15 development patches, separate from the frozen 20-patch evaluation. Reconstructed HoVer-Net scores match every saved historical baseline. No probability maps were available and no settings were tuned during this test.

| Method | Mean biological mIoU | Mean binary nucleus IoU | Improved / worse / unchanged patches (mIoU) |
|---|---:|---:|---|
| HoVer-Net | 0.474933 | 0.746156 | — |
| Stage 1 only: mask refinement, original types propagated | 0.474896 | 0.746096 | 0 / 5 / 10 |
| Stage 2 only: nucleus-type graph, original mask fixed | 0.426633 | 0.746156 | 0 / 5 / 10 |
| Both stages | 0.426603 | 0.746096 | 0 / 7 / 8 |
| Unified six-label reference | 0.472419 | 0.740014 | 2 / 12 / 1 |

- **The single-patch gain did not generalize.** The full two-stage run made 85 wrong → correct changes but 2,347 correct → wrong changes.
- **Most damage came from stage 2.** With the original mask fixed, nucleus reclassification alone made 84 corrections and 2,333 new errors. One wrong nucleus-level decision changes many pixels at once. Stage 2 also uses a 12-pixel nucleus graph, so its effect combines the change in variables with longer-range edges.
- **Background refinement alone did not help:** it changed 19 pixels, made 1 correction and 14 new errors, and slightly reduced binary IoU.
- **The unified reference also worsened on average**, although less than this two-stage configuration. Different cost scales and graph representations prevent attributing that difference solely to unified versus sequential optimization.

Results: [summary](two_stage_dev_trial/summary.json), [per-patch comparisons](two_stage_dev_trial/per_patch.csv). Reproduce with a new output directory:

```bash
python experiments/pannuke_d5/run_two_stage_dev.py --output /tmp/pannuke_two_stage_dev
```

The runner reads selected masks directly from the local Fold 2 zip without extracting its full 7.9 GB mask array. Optimization receives predictions only; ground truth is used for evaluation and preprocessing verification.

## Development check with class-dependent vectors

The matrix was frozen after fitting the separate calibration set, then tested on the same 15 development patches. Stage-1 costs, graph radius and pairwise strength were unchanged.

| Method | Mean biological mIoU |
|---|---:|
| HoVer-Net | 0.474933 |
| Two stages, previous flat unary costs | 0.426603 |
| Two stages, class-dependent unary vectors | 0.432511 |

The class-dependent vectors improve on the flat-cost experiment but still worsen HoVer-Net: 1 patch improves, 6 worsen, 8 are unchanged. There are 85 corrections and 1,961 new errors, versus 2,347 new errors with flat costs. Background-stage binary IoU is unchanged at 0.746096. This does not establish a successful refinement method.

Saved [summary](two_stage_class_costs_dev_trial/summary.json) and [per-patch results](two_stage_class_costs_dev_trial/per_patch.csv). Reproduce with:

```bash
python experiments/pannuke_d5/run_two_stage_dev.py \
  --output /tmp/pannuke_class_costs_dev \
  --reference-results experiments/pannuke_d5/two_stage_dev_trial
```

## Two stages or unified?

These tested two-stage settings should not replace HoVer-Net: they worsen the development result. Two stages make detection and classification costs easier to inspect and let the type stage preserve whole nuclei. Their drawback is irreversible segmentation decisions: type evidence in stage 2 cannot repair the stage-1 mask. A unified model can couple presence and type, but needs meaningful evidence and calibrated costs for all six labels. Keep both as experiments; choose using the same development patches, matched inputs, and separate detection/type metrics. Do not select from this single-class example or tune on the frozen 20-patch evaluation set.

BCSS and historical D5 results are unchanged.
