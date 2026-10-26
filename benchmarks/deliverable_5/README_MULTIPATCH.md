# Deliverable 5 — PanNuke Multi-Patch WCSP Evaluation

## Overview

This directory extends the original Deliverable 5 single-patch PanNuke experiment into a larger multi-patch evaluation.

The original one-patch experiment was useful for verifying that the complete pipeline worked:

1. Run HoVer-Net on a real PanNuke image.
2. Convert predicted nucleus instances into the five WCSP biological classes.
3. Construct a pixel-level WCSP over predicted nucleus pixels.
4. Run strong- and weak-fidelity optimization.
5. Compare the optimized labels against PanNuke ground truth.

The multi-patch experiment was added because a single patch was not enough to determine whether the WCSP consistently improves HoVer-Net predictions.

The goal of this expanded experiment was therefore to answer two questions:

1. Does the current WCSP improve HoVer-Net classifications across multiple PanNuke patches?
2. If the WCSP rarely changes predictions, is this mainly because the unary fidelity penalty is too strong?

---

# Label Mapping

The WCSP uses five biological labels:

| ID  | Class                     |
| --- | ------------------------- |
| 1   | Neoplastic                |
| 2   | Inflammatory              |
| 3   | Connective                |
| 4   | Dead                      |
| 5   | Non-neoplastic epithelial |

Background pixels use label `0` and are inactive.

Only pixels belonging to a HoVer-Net predicted nucleus are treated as WCSP variables.

Ground truth is used only during evaluation and is never passed to the optimizer.

---

# PanNuke / HoVer-Net Preparation

For every selected PanNuke patch, `prepare_full_patch.py` creates the following files:

```text
hovernet_labels_164.npy
active_mask_164.npy
pannuke_gt_labels_164.npy
hovernet_instance_map_164.npy
pannuke_d5_patch_package.npz
metadata.json
```

The HoVer-Net valid prediction region is `164 x 164`, corresponding to the central region:

```text
[46:210, 46:210]
```

of the original `256 x 256` PanNuke image.

Each HoVer-Net nucleus instance is assigned its predicted biological class, and that class is copied to all pixels belonging to the instance.

Pixels not belonging to a mapped HoVer-Net nucleus remain inactive.

---

# WCSP Formulation

For each active pixel, the WCSP domain is:

```text
{1, 2, 3, 4, 5}
```

The unary cost is:

```text
0       if the WCSP keeps the HoVer-Net label
lambda  if the WCSP changes the HoVer-Net label
```

The pairwise cost uses the project's hand-written biological compatibility table:

```text
pairwise_cost(i, j)
    = HANDWRITTEN[label_i, label_j] / distance(i, j)^eta
```

with:

```text
eta = 1
```

The same five spatial neighborhood types used in Deliverable 4 are retained:

```text
squared distance = 1
squared distance = 2
squared distance = 4
squared distance = 5
squared distance = 8
```

Pairs touching inactive background pixels are omitted.

Optimization uses:

```text
BlockCoordinateDescent(ICMSolver)
```

with four corner blocks and a central cross.

---

# Strong Fidelity Check

A strong-fidelity run is retained as a correctness check.

The strong lambda value is chosen above the maximum possible pairwise saving from changing a pixel.

Therefore the optimizer should return the original HoVer-Net labeling exactly.

Across the multi-patch experiment, strong fidelity preserved the HoVer-Net labels as expected.

This confirms that the solver respects the unary fidelity construction.

---

# Initial 20-Patch Evaluation

A diverse set of 20 PanNuke Fold 2 patches was selected.

The set included tissue types such as:

```text
Bladder
Liver
Stomach
Breast
Colon
Lung
Uterus
Thyroid
HeadNeck
Bile-duct
Testis
Esophagus
Adrenal gland
```

The patches contained at least three ground-truth biological classes, with several containing four classes.

The selected patch list is stored in:

```text
pannuke_multi_patch_selection.csv
```

Prepared patches are stored under:

```text
pannuke_multi_patch/
```

---

# 20-Patch WCSP Results

The initial weak-fidelity experiment used:

```text
lambda = 1
```

Across the 20 evaluation patches:

```text
13 / 20 patches had zero initial pairwise energy.

19 / 20 patches had no WCSP label changes.

1 / 20 patches changed.

Total changed pixels = 2.
```

The only patch with changes was:

```text
image_001409
Tissue: Bile-duct
```

For this patch:

```text
WCSP changed pixels: 2

wrong -> correct:          0
correct -> wrong:          2
wrong -> different wrong:  0
GT background changes:     0
```

Both incorrect changes were:

```text
Non-neoplastic epithelial -> Connective
```

The WCSP reduced its optimization objective but slightly worsened the ground-truth metrics.

For this patch:

```text
Class agreement on overlap:
0.8147 -> 0.8143

mIoU:
0.6190 -> 0.6185
```

---

# Aggregate 20-Patch Evaluation

Across all 20 patches:

```text
Total WCSP-changed pixels:       2

Wrong -> correct:                0
Correct -> wrong:                2
Wrong -> different wrong:        0
Changes on GT background:        0
```

Patch-level mIoU outcomes were:

```text
Improved:   0
Worsened:   1
Unchanged: 19
```

The aggregate error counts were:

```text
Missed GT nucleus pixels:          12,623
False-positive predicted pixels:   16,574

HoVer-Net classification errors:   22,249
WCSP classification errors:        22,251
```

Therefore, under the current formulation and `lambda = 1`, the WCSP produced essentially no improvement over the HoVer-Net baseline.

Importantly, this does not imply that WCSP optimization itself is incorrect.

It shows that the current PanNuke representation and cost formulation provide very limited useful refinement.

---

# Why a Lambda Sweep Was Added

One possible explanation for the limited number of changes was that:

```text
lambda = 1
```

might still be too strong.

If the unary fidelity penalty prevented useful pairwise corrections, reducing lambda should allow the WCSP to make more changes and potentially improve accuracy.

To test this without tuning on the 20 evaluation patches, a separate development set was created.

---

# Lambda Development Set

A separate set of 15 unused PanNuke patches was selected.

The development selection is stored in:

```text
pannuke_lambda_dev_selection.csv
```

These patches were not part of the previous 20-patch evaluation set.

The following values were tested:

```text
lambda = 1.00
lambda = 0.50
lambda = 0.25
lambda = 0.00
```

Using `lambda = 0` completely removes the unary penalty for changing the HoVer-Net label.

This makes it possible to determine whether unary fidelity is the primary reason the WCSP remains close to the original predictions.

---

# Lambda Sweep — Optimization Activity

Across the 15 development patches:

| Lambda | Total Changed Pixels | Patches With Changes |
| -----: | -------------------: | -------------------: |
|   1.00 |                   83 |               4 / 15 |
|   0.50 |                   85 |               4 / 15 |
|   0.25 |                   94 |               5 / 15 |
|   0.00 |                  145 |               5 / 15 |

Reducing lambda therefore increased the number of WCSP changes, but only moderately.

Even when:

```text
lambda = 0
```

and there was no fidelity penalty at all:

```text
10 / 15 patches still had no WCSP changes.
```

This suggests that unary fidelity is not the primary reason the optimizer is inactive on many patches.

---

# Lambda Sweep — Ground-Truth Evaluation

The changes produced at each lambda were evaluated against PanNuke ground truth.

## lambda = 1.00

```text
Changed pixels:          83
Wrong -> correct:         1
Correct -> wrong:        28
Net corrections:        -27

Patches improved:         0
Patches worsened:         3

Mean delta mIoU:
-0.000539
```

## lambda = 0.50

```text
Changed pixels:          85
Wrong -> correct:         1
Correct -> wrong:        27
Net corrections:        -26

Patches improved:         0
Patches worsened:         3

Mean delta mIoU:
-0.000528
```

## lambda = 0.25

```text
Changed pixels:          94
Wrong -> correct:         2
Correct -> wrong:        27
Net corrections:        -25

Patches improved:         1
Patches worsened:         3

Mean delta mIoU:
-0.000530
```

## lambda = 0.00

```text
Changed pixels:         145
Wrong -> correct:         2
Correct -> wrong:        61
Net corrections:        -59

Patches improved:         1
Patches worsened:         3

Mean delta mIoU:
-0.001034
```

---

# Main Finding

Reducing the fidelity weight does increase the number of label changes, but the additional changes are generally not beneficial.

The clearest result occurs at:

```text
lambda = 0
```

where the optimizer is completely free to ignore the original HoVer-Net class prediction.

Even under this setting:

```text
Only 2 incorrect pixels were corrected.

61 previously correct pixels became incorrect.
```

Therefore:

```text
Net corrections = -59
```

and the mean mIoU decreased.

This indicates that excessive unary fidelity is not the main limitation of the current PanNuke WCSP.

---

# Interpretation

The experiments suggest that the main limitation is more likely associated with the current representation and/or pairwise compatibility model.

In the current formulation:

1. Each HoVer-Net nucleus already has one uniform biological label.

2. WCSP variables are individual active pixels.

3. Background pixels are inactive.

4. Pairwise constraints only connect nearby active pixels.

As a result, many neighboring nuclei are separated by inactive background pixels and do not directly interact in the WCSP graph.

Inside a single predicted nucleus, neighboring pixels usually already share the same class, giving them zero pairwise disagreement cost.

Therefore many patches either:

```text
have zero pairwise conflict
```

or:

```text
contain too little useful pairwise structure to correct the original classification.
```

When the pairwise model does cause changes, those changes often do not align with the ground-truth biological class.

---

# Conclusion

The Deliverable 5 PanNuke experiments show that the current pixel-level WCSP does not consistently improve HoVer-Net classifications.

Across the 20-patch evaluation set:

```text
0 / 20 patches improved in mIoU.

19 / 20 patches were unchanged.

Only 2 pixels changed in total.

Both changes introduced new errors.
```

A separate 15-patch lambda sweep showed that reducing or completely removing unary fidelity does not solve the problem.

At `lambda = 0`, WCSP activity increased, but the additional changes were substantially more likely to introduce errors than correct them.

Therefore the current evidence suggests that the principal limitation is not simply the fidelity parameter.

Instead, the pixel-level interaction graph and/or hand-designed pairwise biological cost table appear insufficient for reliable refinement of HoVer-Net PanNuke predictions.

---

# Future Work

A more promising future formulation would operate at the nucleus level rather than the individual pixel level.

For example:

```text
one HoVer-Net nucleus = one WCSP variable
```

with the domain:

```text
{Neoplastic,
 Inflammatory,
 Connective,
 Dead,
 Non-neoplastic epithelial}
```

Pairwise constraints could then connect nearby nuclei using:

```text
centroid distance
boundary distance
nearest-neighbor relationships
or tissue-level spatial context
```

This would allow biologically related nuclei to interact even when background pixels separate their masks.

This nucleus-level redesign is proposed as future work and was not implemented as part of the current Deliverable 5 experiment.

---

# Main Scripts

## Input preparation

```text
pannuke_preparation/prepare_full_patch.py
```

Converts one PanNuke/HoVer-Net patch into the arrays required by the WCSP and evaluation pipeline.

---

## Multi-patch selection

```text
select_pannuke_patches.py
make_pannuke_multipatch_set.py
```

Used to identify class-diverse PanNuke patches and construct the 20-patch evaluation set.

---

## Multi-patch preparation

```text
run_prepare_multipatch.py
```

Prepares all selected evaluation patches.

---

## WCSP solver

```text
pannuke_refinement/run_wcsp.py
```

Runs:

```text
unary-only
strong-fidelity
weak-fidelity
```

WCSP settings.

The script was generalized to accept:

```text
--source
--output
--weak-lambda
```

so it can be reused across multiple patches and lambda values.

---

## Multi-patch WCSP execution

```text
run_wcsp_multipatch.py
```

Runs the WCSP over the 20-patch evaluation set.

---

## Ground-truth evaluation

```text
pannuke_refinement/evaluate.py
```

Compares HoVer-Net and WCSP labels against PanNuke ground truth.

It reports:

```text
wrong -> correct
correct -> wrong
wrong -> different wrong
changes on GT background
class agreement
pixel accuracy
mIoU
detection errors
classification errors
```

---

## Multi-patch evaluation

```text
run_evaluate_multipatch.py
```

Aggregates the evaluation results across the 20-patch evaluation set.

Output:

```text
pannuke_multi_patch_evaluation_summary.csv
```

---

## Lambda development set

```text
make_pannuke_lambda_dev_set.py
run_prepare_lambda_dev.py
```

Creates and prepares a separate 15-patch set used for fidelity analysis.

---

## Lambda sweep

```text
run_lambda_dev_sweep.py
```

Runs:

```text
lambda = 1.0
lambda = 0.5
lambda = 0.25
lambda = 0.0
```

over all 15 development patches.

Output:

```text
pannuke_lambda_dev_wcsp_summary.csv
```

---

## Lambda evaluation

```text
run_lambda_dev_evaluate.py
```

Evaluates all 60 development-set WCSP outputs against ground truth.

Output:

```text
pannuke_lambda_dev_evaluation_summary.csv
```

---

# Summary

The full Deliverable 5 PanNuke workflow now includes:

```text
single-patch integration test
        ↓
20-patch evaluation
        ↓
ground-truth comparison
        ↓
separate 15-patch development set
        ↓
lambda sweep
        ↓
ground-truth lambda evaluation
        ↓
representation-level diagnosis
```

The main result is that lowering fidelity does not produce reliable biological corrections.

The current pixel-level WCSP should therefore be treated as a validated experimental formulation whose limitations have been identified, rather than further tuned on the same evaluation data.
