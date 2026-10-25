# Deliverable 5 — PanNuke Full-Patch Input Preparation

## Objective

This folder prepares a real PanNuke HoVer-Net prediction for the Deliverable 5 WCSP pipeline. The goal is to take one complete HoVer-Net prediction region, convert the predicted nucleus instances into pixel-level biological labels, align the corresponding PanNuke ground truth, and provide reproducible inputs for the WCSP optimization and evaluation stages.

The selected example is **PanNuke Fold 2, `image_00748`**.

The intended pipeline is:

```text
PanNuke image
    ↓
HoVer-Net prediction
    ↓
164×164 predicted pixel-label map
    ↓
WCSP refinement
    ↓
refined labels
    ↓
comparison with aligned PanNuke ground truth
```

Ground truth is used only for evaluation and must not be used by the WCSP solver when selecting labels.

---

## Aakanksha — Full-Patch Input Preparation

### Source prediction

The input comes from the existing `hovernet_fast-pannuke` inference performed on PanNuke Fold 2.

For `image_00748`, HoVer-Net produced:

- **47 predicted nuclei**
- a **164×164 instance segmentation map**
- predicted nucleus classes stored in the Fold 2 prediction CSV
- centroids used to associate each predicted nucleus with its corresponding instance ID

All 47 predicted nuclei were successfully mapped from the prediction metadata to their instance-map regions.

### Conversion to pixel labels

HoVer-Net's instance map stores a unique integer ID for each detected nucleus. The corresponding nucleus class from the prediction CSV was assigned to every pixel belonging to that instance.

The WCSP label mapping is:

|  ID | Biological class          |
| --: | ------------------------- |
|   0 | Background / inactive     |
|   1 | Neoplastic                |
|   2 | Inflammatory              |
|   3 | Connective                |
|   4 | Dead                      |
|   5 | Non-neoplastic epithelial |

A pixel with label `0` is **not interpreted as healthy tissue**. It means that the pixel is not part of a usable HoVer-Net nucleus prediction and is therefore excluded from the active WCSP variables.

### Alignment with PanNuke ground truth

The original PanNuke patch is **256×256**, while the saved fast HoVer-Net prediction corresponds to the central **164×164** region.

The valid HoVer-Net region therefore corresponds to:

```text
x = 46 ... 209
y = 46 ... 209
```

in the original PanNuke patch.

The corresponding PanNuke ground-truth mask was cropped to exactly this same 164×164 region before evaluation.

### Current diagnostics

For `image_00748`:

| Quantity                                          |  Value |
| ------------------------------------------------- | -----: |
| Predicted nuclei                                  |     47 |
| Active HoVer-Net pixels                           | 10,726 |
| Inactive/background pixels                        | 16,170 |
| Ground-truth nucleus pixels                       |  9,250 |
| Prediction/GT overlapping nucleus pixels          |  8,229 |
| Predicted nucleus pixels where GT is background   |  2,497 |
| GT nucleus pixels missed by predicted active mask |  1,021 |
| Class agreement on overlapping nucleus pixels     | 0.7737 |

The **0.7737 class agreement is not overall pixel accuracy**. It is the fraction of overlapping nucleus pixels where both HoVer-Net and PanNuke assign a nucleus and their biological class labels agree.

The 1,021 ground-truth nucleus pixels that fall outside the HoVer-Net active mask represent an important upstream limitation: these pixels do not become WCSP variables and therefore cannot be recovered through label refinement alone.

---

## Generated Files

### `hovernet_labels_164.npy`

Shape:

```text
(164, 164)
```

This is the main WCSP input.

Each active pixel contains one of the biological labels `1–5`.

Pixels with value `0` are inactive and should not be treated as WCSP variables.

---

### `active_mask_164.npy`

Boolean mask with shape:

```text
(164, 164)
```

Use:

```python
active_mask[y, x] == True
```

to determine whether a pixel participates in the WCSP.

There are currently:

```text
10,726 active pixels
```

for `image_00748`.

---

### `hovernet_instance_map_164.npy`

The original HoVer-Net instance segmentation map.

Each positive integer corresponds to one detected nucleus.

This should be retained so results can later be analyzed at the nucleus level rather than only at the pixel level.

---

### `pannuke_gt_labels_164.npy`

Aligned PanNuke ground-truth class labels for the same 164×164 region.

**Important:** this file is for evaluation only.

It must not be used to choose WCSP labels or tune the result for this test patch.

---

### `pannuke_d5_patch_package.npz`

Convenience bundle containing:

```text
predicted_labels
active_mask
ground_truth_labels
instance_map
```

This can be loaded using:

```python
import numpy as np

data = np.load("pannuke_d5_patch_package.npz")

initial_labels = data["predicted_labels"]
active_mask = data["active_mask"]
ground_truth = data["ground_truth_labels"]
instance_map = data["instance_map"]
```

Again, `ground_truth` should be ignored during optimization and used only afterward for evaluation.

---

### `metadata.json`

Contains:

- dataset and image ID
- coordinate information
- label dictionary
- active/background policy
- number of mapped nuclei
- prediction and ground-truth pixel counts
- overlap statistics
- source file locations

---

### `prepare_full_patch.py`

Reproduces the complete preparation pipeline:

```text
saved HoVer-Net instance map
        +
saved HoVer-Net nucleus predictions
        ↓
164×164 predicted class map
        +
active mask
        +
aligned PanNuke ground truth
```

Run from the repository root using:

```bash
conda activate pannuke

python benchmarks/deliverable_5/pannuke_aakanksha/prepare_full_patch.py
```

---

# Instructions for Kashvi — WCSP Optimization

Your input should come from this folder.

Use:

```text
hovernet_labels_164.npy
active_mask_164.npy
```

or load the same arrays from:

```text
pannuke_d5_patch_package.npz
```

## Initial assignment

The starting WCSP label for every active pixel is:

```python
initial_labels[y, x]
```

where:

```python
active_mask[y, x] == True
```

Only active pixels should become WCSP variables.

Pixels where:

```python
active_mask[y, x] == False
```

must remain inactive.

Do **not** convert label `0` into Non-neoplastic epithelial or another biological class.

## Domain

Each active variable should retain the full five-class domain:

```text
{1, 2, 3, 4, 5}
```

corresponding to:

```text
1 = Neoplastic
2 = Inflammatory
3 = Connective
4 = Dead
5 = Non-neoplastic epithelial
```

## D4 behavior to preserve

Deliverable 4 used a unary penalty of:

```text
0     if a pixel keeps its HoVer-Net label
lambda if a pixel changes label
```

and pairwise costs derived from the supplied compatibility matrix divided by pixel distance.

For D5, preserve the same interpretation so the result remains comparable with D4.

## Required runs

At minimum, run:

### 1. Unary-only

Pairwise interactions disabled.

Expected behavior:

```text
returned labels = HoVer-Net labels
```

### 2. Strong fidelity

Use pairwise interactions together with a sufficiently large unary fidelity penalty.

The strong setting must reproduce the HoVer-Net solution.

Do not simply assume that the chosen weight is large enough; document why it is sufficient or verify empirically that all initial labels are preserved.

### 3. Weaker fidelity

Reduce the unary fidelity penalty so that the pairwise biological/spatial costs can alter some labels where doing so lowers the total objective.

## Scalability

Do not enumerate all possible assignments.

D4 contained only eight variables and could enumerate:

```text
5^8 = 390,625
```

assignments.

This full input contains thousands of active pixels, so the scalable/block-based method should be used.

Reuse the existing shared WCSP implementation where possible.

## Required outputs

Please save:

```text
wcsp_strong_labels.npy
wcsp_weak_labels.npy
results.json
```

`results.json` should include at least:

```text
number of WCSP variables
number of unique pairwise edges
unary settings
pairwise settings
initial objective
returned objective
changed-pixel count
runtime
solver status
number of optimization rounds
```

Please do **not** use `pannuke_gt_labels_164.npy` while optimizing.

---

# Instructions for Abhishek — Verification and Ground-Truth Evaluation

Use Aakanksha's:

```text
hovernet_labels_164.npy
active_mask_164.npy
pannuke_gt_labels_164.npy
```

and Kashvi's returned WCSP label maps.

Your role is independent verification and evaluation.

## 1. Verify strong fidelity

Check that:

```text
WCSP strong output == initial HoVer-Net labels
```

for every active pixel.

Report:

```text
changed pixels
initial objective
returned objective
runtime/status
```

The expected changed-pixel count under strong fidelity is:

```text
0
```

if the fidelity bound is implemented correctly.

## 2. Evaluate the weaker-fidelity result

Compare three aligned arrays:

```text
HoVer-Net initial labels
WCSP refined labels
PanNuke ground truth
```

The most important analysis is not simply:

```text
How many pixels changed?
```

Instead, classify changed pixels as:

```text
wrong → correct
correct → wrong
wrong → different wrong
```

This determines whether the WCSP is improving the model rather than merely smoothing its output.

## 3. Report before/after performance

At minimum report:

```text
HoVer-Net baseline class agreement
WCSP class agreement
difference
number of changed pixels
```

If appropriate, also compute:

- pixel accuracy
- class-wise IoU
- mean IoU
- confusion matrix

However, clearly distinguish evaluation on the active HoVer-Net region from evaluation over the complete ground-truth nucleus region.

## 4. Track upstream detection error separately

Aakanksha's preparation found:

```text
1,021 ground-truth nucleus pixels
```

that are not part of the HoVer-Net active mask.

Those pixels cannot be corrected by the WCSP because they were never included as variables.

Keep this separate from classification/refinement error.

The research interpretation should therefore distinguish:

```text
Detection error:
HoVer-Net failed to provide a usable nucleus/pixel.

Classification error:
HoVer-Net provided a nucleus/pixel but predicted the wrong class.

WCSP correction:
A classification error was changed to the ground-truth class.
```

## 5. Do not use ground truth during optimization

Ground truth should be used only after Kashvi returns the final labels.

This maintains a clean experimental pipeline:

```text
HoVer-Net
    ↓
WCSP
    ↓
final prediction
    ↓
ground-truth evaluation
```

and avoids leaking the evaluation labels into the optimizer.

---

## Deliverable 5 PanNuke Goal

The goal of this stage is not yet to prove that the WCSP consistently improves HoVer-Net across PanNuke.

The immediate goal is to demonstrate a working end-to-end pipeline on a real full prediction region:

```text
PanNuke image
    ↓
HoVer-Net
    ↓
pixel-level model prediction
    ↓
WCSP refinement
    ↓
ground-truth comparison
```

If this pipeline works for `image_00748`, the next stage is to scale the same experiment to multiple patches and evaluate whether the biological constraints consistently improve held-out performance rather than merely smoothing individual examples.
