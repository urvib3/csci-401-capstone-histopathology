# Deliverable 5 — WCSP Refinement of Classifier Predictions (PanNuke + BCSS)

We take a trained classifier's per-pixel predictions and refine them with a **weighted constraint satisfaction problem (WCSP)** that encodes which tissue/cell types should and shouldn't sit next to each other. We test this on two datasets:

| | **PanNuke** (nuclei) | **BCSS** (tissue regions) |
|---|---|---|
| Classifier | HoVer-Net (nucleus instances + class) | TIAToolbox U-Net (per-pixel tissue class) |
| WCSP grid | 164 × 164 HoVer-Net region; only nucleus pixels are variables | central 256 × 256 of each 512 tile; every pixel is a variable |
| Labels | neoplastic, inflammatory, connective, dead, non-neoplastic epithelial | tumor, inflammatory, stroma, necrosis, other |
| Code | shared [`wcsp/`](../wcsp/) solver + [`experiments/pannuke_d5/`](../experiments/pannuke_d5/) | shared [`wcsp/`](../wcsp/) package + [`experiments/bcss_wcsp/`](../experiments/bcss_wcsp/) |

**Headline.** The WCSP only reliably fixes **boundaries**, never whole wrong regions.
- **BCSS:** a small but real gain, mIoU 0.6489 → 0.6507 (+0.0019, 95% CI [+0.0010, +0.0027]) on 2,124 held-out-hospital tiles.
- **PanNuke:** almost no changes, and the few changes it makes are mostly wrong.

---

## 1. The WCSP (shared)

![wcsp](../experiments/bcss_wcsp/figures/wcsp_diagram.png)

- **Variables:** pixels. **Domain:** the 5 class labels.
- **Objective:** find the labeling with the lowest total cost = Σ unary + Σ pairwise.
- **Pairwise cost (the biology rules):**
  - Applies to every pixel pair up to 2√2 px apart: 24 neighbors, 5 distance types (left panel).
  - Cost = b[label, label′] / distance. Closer neighbors count more; same label is free.
  - **b** is the class-compatibility table. PanNuke uses the handwritten table from the project notes. BCSS tests both the handwritten table and one **derived from the BCSS training labels** (each cell: handwritten / data-derived; details in §3):

**PanNuke (handwritten)**

| | Neoplastic | Inflammatory | Connective | Dead | Non-neoplastic epithelial |
|---|---|---|---|---|---|
| **Neoplastic** | 0 | 2 | 1 | 1 | 9 |
| **Inflammatory** | | 0 | 2 | 1 | 8 |
| **Connective** | | | 0 | 5 | 1 |
| **Dead** | | | | 0 | 10 |
| **Non-neoplastic epithelial** | | | | | 0 |

**BCSS (handwritten / data-derived)**

| | Tumor | Inflammatory | Stroma | Necrosis | Other |
|---|---|---|---|---|---|
| **Tumor** | 0 | 2 / 5.2 | 1 / 3.0 | 1 / 4.1 | 9 / 4.5 |
| **Inflammatory** | | 0 | 2 / 2.2 | 1 / 6.9 | 8 / 3.2 |
| **Stroma** | | | 0 | 5 / 3.6 | 1 / 2.3 |
| **Necrosis** | | | | 0 | 10 / 4.9 |
| **Other** | | | | | 0 |

  Symmetric, so only the upper triangle is shown; same label (diagonal) is free.

- **Unary cost (trust the classifier).** This is where the two datasets differ:

  | | PanNuke | BCSS |
  |---|---|---|
  | Unary | **λ if the pixel changes from HoVer-Net's label, else 0** | **−log p(label)** from the U-Net (p = 0.9 → 0.1, p = 0.01 → 4.6) |
  | Uses classifier certainty? | No: HoVer-Net's saved output has labels only | **Yes:** unsure pixels are cheap to change |

- **The trade-off knob λ (same idea on both datasets):** λ balances *trust the classifier* against *follow the rules*, and only that balance matters. PanNuke writes it as λ · unary + pairwise; BCSS writes it as unary + λ · pairwise. Those are the same thing with λ inverted: a small λ on PanNuke is a large λ on BCSS. So on PanNuke a *lower* λ means stronger rules, and on BCSS a *higher* λ does.

- **Solver (both datasets):** block scheme + ICM.
  - Block scheme (right panel): solve the four corner blocks with the cross fixed, then the cross with the corners fixed; repeat. Blocks are 100/56/100 px on BCSS and 60/44/60 px on PanNuke.
  - ICM inside each block: switch each pixel to its cheapest label until nothing changes. It's greedy and fast, and finds a local optimum.
  - BCSS also compares **toulbar2**, an exact WCSP solver, on the same problem (§3).
- **Intuition:** a pixel's label changes only if the rule savings beat its unary cost. Inside a region (or nucleus) every neighbor agrees, so there's nothing to gain. Only **boundary** pixels ever change.


### Hyperparameters

| Knob | What it controls | PanNuke | BCSS |
|---|---|---|---|
| **λ** | classifier vs. rules trade-off | 1 (also 0–0.5, and 139 as the strong-fidelity check) | 0.5–32 swept on dev; **8** reported |
| **b** | pairwise cost for each pair of classes | handwritten | handwritten **or** data-derived (PMI) |
| **unary** | how the classifier's opinion is scored | λ · [label changed] | −log p, raw softmax (temperature 1) |
| **neighborhood** | which pixel pairs interact | 5 distance types, up to 2√2 px (24 neighbors) | same |
| **η** | distance falloff, b / dᵑ | 1 | 1 |
| **f: β, σ** | image term: extra cost for a label change between similar colors | off | off (β = 1, 2, 4 tested; hurt) |
| **solver** | search inside each block | ICM | ICM (toulbar2 compared) |
| **blocks** | corner size / cross width | 60 / 44 px | 100 / 56 px |
| **rounds, time limit** | how long to iterate | until no change | ≤ 4 rounds; toulbar2 20 s per block |

All of these are set in one YAML config, e.g. `experiments/bcss_wcsp/configs/bcss_unet.yaml`, or per run with `--set problem.lam=8`.

---

---

## 2. PanNuke

### Pipeline and experiments

1. HoVer-Net runs on a 256 × 256 PanNuke image. Its valid output is the central 164 × 164.
2. Each predicted nucleus's class is copied to all its pixels. Background pixels are inactive, not variables.
3. WCSP on the nucleus pixels, then compared with the aligned PanNuke ground truth (used only for evaluation).
4. **Sweeps:**
   - **Single patch** (`image_00748`): λ = 1 (weak fidelity) and λ = 139 (strong fidelity; provably no changes).
   - **20 diverse evaluation patches** at λ = 1.
   - **λ sweep** on 15 separate dev patches: λ = 1, 0.5, 0.25, 0. This asks whether the fidelity penalty is what blocks useful changes.

**Metrics:** class agreement on overlapping nucleus pixels, pixel accuracy and IoU, and counts of wrong → correct vs. correct → wrong changes.

### Results

| Experiment | Setting | HoVer-Net | + WCSP | Verdict |
|---|---|---|---|---|
| 1 patch | λ = 1 | class agreement 0.7737 | 0.7730 | 13 px changed: 0 fixed, 6 new errors |
| 1 patch | λ = 139 (strong fidelity) | — | identical | 0 changes, as the bound guarantees |
| 20 patches | λ = 1 | — | — | 2 px changed in total, both new errors; 19/20 unchanged |
| 15 dev patches | λ = 0 (no fidelity) | — | mean Δ mIoU −0.0010 | 2 fixed, 61 broken |

**Why the WCSP doesn't help on PanNuke:**
- Nuclei are separated by inactive background, so neighboring nuclei barely interact.
- Each nucleus already has one uniform label, so there's little pairwise conflict inside it.
- The flat unary can't tell confident pixels from unsure ones.
- The changes that do happen don't follow the true classes.

<details>
<summary><b>PanNuke: single full patch (image_00748)</b></summary>

Ground truth is used only for evaluation; the WCSP solver never reads it. The prepared inputs are committed, so the WCSP and evaluation steps run without the PanNuke data:

```bash
conda activate pannuke
python experiments/pannuke_d5/pannuke_preparation/prepare_full_patch.py   # needs the Fold 2 data
python experiments/pannuke_d5/pannuke_refinement/run_wcsp.py
python experiments/pannuke_d5/pannuke_refinement/evaluate.py
```

#### Input preparation 

##### Source prediction

The input comes from the existing `hovernet_fast-pannuke` inference performed on PanNuke Fold 2.

For `image_00748`, HoVer-Net produced:

- **47 predicted nuclei**
- a **164×164 instance segmentation map**
- predicted nucleus classes stored in the Fold 2 prediction CSV
- centroids used to associate each predicted nucleus with its corresponding instance ID

All 47 predicted nuclei were successfully mapped from the prediction metadata to their instance-map regions.

##### Conversion to pixel labels

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

##### Alignment with PanNuke ground truth

The original PanNuke patch is **256×256**, while the saved fast HoVer-Net prediction corresponds to the central **164×164** region:

```text
x = 46 ... 209
y = 46 ... 209
```

The corresponding PanNuke ground-truth mask was cropped to exactly this same 164×164 region before evaluation.

##### Diagnostics

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

##### Files in [`experiments/pannuke_d5/pannuke_preparation/`](../experiments/pannuke_d5/pannuke_preparation/)

| File | Contents |
| --- | --- |
| `hovernet_labels_164.npy` | (164, 164) main WCSP input. Active pixels carry labels 1–5, inactive pixels are 0. |
| `active_mask_164.npy` | (164, 164) bool. `active_mask[y, x] == True` means the pixel is a WCSP variable (10,726 pixels). |
| `hovernet_instance_map_164.npy` | Original HoVer-Net instance map; each positive integer is one nucleus. Kept for nucleus-level analysis. |
| `pannuke_gt_labels_164.npy` | Aligned PanNuke ground-truth class labels. **Evaluation only.** |
| `pannuke_d5_patch_package.npz` | Bundle of `predicted_labels`, `active_mask`, `ground_truth_labels`, `instance_map`. |
| `metadata.json` | Dataset/image ID, coordinates, label dictionary, active/background policy, counts, overlap statistics, source paths. |
| `prepare_full_patch.py` | Reproduces the files above from the saved HoVer-Net instance map + prediction CSV and the Fold 2 `masks.npy`. |

HoVer-Net instance map
        +
HoVer-Net nucleus class predictions
        ->
hovernet_labels_164.npy
        ->
active_mask_164.npy = hovernet_labels > 0
        ->
WCSP gets:
  1. label map
  2. active mask

```python
import numpy as np

data = np.load("experiments/pannuke_d5/pannuke_preparation/pannuke_d5_patch_package.npz")

initial_labels = data["predicted_labels"]
active_mask = data["active_mask"]
ground_truth = data["ground_truth_labels"]   # ignore during optimization
instance_map = data["instance_map"]
```

---

#### WCSP run 

`run_wcsp.py` only reads `hovernet_labels_164.npy` and `active_mask_164.npy`.

Same setup as D4 so the results are comparable:

- Variables: the 10,726 active pixels, each with domain {1, ..., 5}. Inactive pixels stay 0.
- Unary: λ if a pixel changes from its HoVer-Net label, 0 if it keeps it.
- Pairwise: the handout cost table divided by distance, over all five neighbor types (up to 2√2 away). Pairs touching an inactive pixel are left out. That gives 111,755 unique edges.
- Solver: the shared `wcsp/` block scheme with ICM inside each block. The default 100x100 blocks don't fit a 164x164 image, so it uses 60x60 corners and a 44 px cross. No assignments are enumerated.

| Run | λ | Initial objective | Returned objective | Changed pixels | Rounds | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unary-only | 1 | 0.00 | 0.00 | 0 | 1 | 1.8 s |
| Strong fidelity | 139 | 772.81 | 772.81 | 0 | 1 | 1.8 s |
| Weaker fidelity | 1 | 772.81 | 723.15 | 13 | 2 | 3.4 s |

Why λ = 139 is strong enough: a pixel has at most 24 neighbors, and the sum of 1/d over them is 13.82. The biggest table cost is 10, so changing one pixel can save at most 138.2 in pairwise cost. Changing any k pixels saves at most 138.2k but costs λk, so with λ > 138.2 no change can lower the objective. This bound is actually reached on this mask, since some pixels have all 24 neighbors active. The run also confirms it empirically: 0 changes.

ICM is greedy, so the results are local optima, not proven global ones. For every run, no single pixel can lower the objective by changing its label, and solving the whole image as one block gives the same weak result (723.1453).

---

#### Ground-truth evaluation 

`evaluate.py` is the only script that opens `pannuke_gt_labels_164.npy`, and it runs after the WCSP labels are saved.

The ground truth for this patch only has one class: all 9,250 nucleus pixels are Inflammatory. I checked this against the raw `masks.npy` channels, so it is not an alignment bug.

Strong fidelity output is identical to HoVer-Net on every pixel.

The 13 pixels changed in the weaker fidelity run:

| Change | Pixels | What changed |
| --- | ---: | --- |
| Wrong → correct | 0 | |
| Correct → wrong | 6 | Inflammatory → Connective |
| Wrong → different wrong | 0 | |
| Ground truth is background | 7 | 5 Connective → Inflammatory, 2 Inflammatory → Connective |

Before and after:

| Metric | HoVer-Net | WCSP (λ = 1) | Difference |
| --- | ---: | ---: | ---: |
| Class agreement on overlap (8,229 px) | 0.7737 | 0.7730 | −0.0007 |
| Accuracy on HoVer-Net active region | 0.5936 | 0.5930 | −0.0006 |
| Accuracy on ground-truth nucleus region | 0.6883 | 0.6877 | −0.0006 |
| Pixel accuracy, full patch | 0.8000 | 0.7997 | −0.0002 |
| Inflammatory IoU | 0.6399 | 0.6391 | −0.0008 |

Error breakdown:

| Error | Pixels | Can the WCSP fix it? |
| --- | ---: | --- |
| Detection: ground-truth nucleus missed by HoVer-Net | 1,021 | No, never a variable |
| Detection: HoVer-Net nucleus where ground truth is background | 2,497 | No, the WCSP can't relabel to background |
| Classification: wrong class on overlap, HoVer-Net | 1,862 | Yes |
| Classification: wrong class on overlap, after WCSP | 1,868 | |

What we see:

- The pipeline works end to end on a real full prediction: HoVer-Net → WCSP → ground-truth comparison.
- Strong fidelity reproduces HoVer-Net exactly, as expected.
- At λ = 1 the WCSP lowers the objective but makes accuracy slightly worse. It made 0 corrections and 6 new errors. A lower objective only means the labels fit the cost table better, not that they are more correct.
- Almost all of HoVer-Net's class errors are whole nuclei labeled Connective (3,231 px) when they are Inflammatory. The WCSP can't fix a whole wrong nucleus because every pixel inside it agrees with its neighbors, so it only moves pixels on the edges where two nuclei touch.
- Same thing we saw on BCSS: the pairwise terms only reach 2 pixels, so they only change boundaries.
- This is one patch with one ground-truth class, so it doesn't say much about PanNuke in general. The next step is running more patches. Making the decision per nucleus (using the instance map) instead of per pixel would also probably fit PanNuke better.

How it was evaluated:

- Overlap = pixels that are nucleus in both HoVer-Net and the ground truth. This is the same 0.7737 class agreement from Section 1, so the evaluation matches the preparation.
- Active region accuracy counts ground-truth background as wrong. Ground-truth nucleus accuracy counts missed pixels as wrong.
- IoU is only reported for Inflammatory since it is the only class in the ground truth. The other predicted classes have IoU 0 both before and after, and Dead is never predicted.
- λ = 1 was taken from D4 and not tuned on this patch, since tuning it here would use the ground truth.

##### Files in [`experiments/pannuke_d5/pannuke_refinement/`](../experiments/pannuke_d5/pannuke_refinement/)

| File | Contents |
| --- | --- |
| `run_wcsp.py`, `results.json` | WCSP runs, settings, objectives, energy trace per round |
| `wcsp_strong_labels.npy`, `wcsp_weak_labels.npy` | Returned 164x164 label maps (0 = inactive) |
| `evaluate.py`, `evaluation.json` | All metrics above plus confusion matrices |

</details>

<details>
<summary><b>PanNuke: 20-patch evaluation and λ sweep</b></summary>

**20 diverse Fold 2 patches, λ = 1:**
- 13/20 patches have zero initial pairwise cost.
- 19/20 are unchanged; only 2 pixels change in total, both new errors (Non-neoplastic epithelial → Connective).
- mIoU: 0 improved, 1 worsened, 19 unchanged.
- The errors are almost all upstream:
  - 12,623 ground-truth nucleus pixels missed;
  - 16,574 predicted pixels where the ground truth is background;
  - 22,249 classification errors before the WCSP and 22,251 after.

**λ sweep on 15 separate dev patches** (does the fidelity penalty block useful changes?):

| λ | Changed px | Patches changed | Wrong → correct | Correct → wrong | Mean Δ mIoU |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 83 | 4/15 | 1 | 28 | −0.00054 |
| 0.50 | 85 | 4/15 | 1 | 27 | −0.00053 |
| 0.25 | 94 | 5/15 | 2 | 27 | −0.00053 |
| 0.00 | 145 | 5/15 | 2 | 61 | −0.00103 |

- **Fidelity isn't the bottleneck.** Even with no unary penalty (λ = 0), 10/15 patches don't change at all, and the changes that do happen break far more than they fix.
- **The limit is the representation and the pairwise table:** pixel variables, inactive background between nuclei, and hand-set b values.
- Full write-up, patch selection, scripts: [deliverable_5_pannuke_multipatch.md](deliverable_5_pannuke_multipatch.md).

</details>

---

## 3. BCSS

### Pipeline and experiments

![pipeline](../experiments/bcss_wcsp/figures/pipeline.png)

1. **Input:** a 512 × 512 tile. **U-Net:** outputs 5 class probabilities per pixel at half resolution (256 × 256), covering the whole tile.
2. **Upsample ×2 to 512 × 512:** each class probability map is upsampled bilinearly, and a pixel's class is the **argmax** of its 5 probabilities. We upsample probabilities, not labels, so the WCSP knows how confident the U-Net is at every pixel.
3. **Keep the central 256 × 256:** the outer ring is context only, since predictions near the edge see less tissue.
4. **WCSP** on all 65,536 crop pixels, giving 256 × 256 refined labels, scored against the ground truth for the same crop.
5. **Splits:** settings are chosen on **dev** (every 8th training region, 185 tiles). Results are reported on **val** (held-out hospitals, 2,124 labeled tiles).
6. **Sweeps, each step answering one question:**
   1. Rule strength λ ∈ {0.5 … 32} × b table (handwritten vs. data-derived), on dev.
   2. Image-edge term f (β = 1, 2, 4), on dev.
   3. Chosen settings on val.
   4. ICM vs. toulbar2 on val.

**Metrics:** pixel accuracy and mIoU (per-class IoU averaged over the 5 classes), with a 95% CI from resampling whole tissue regions.

### Results

| Experiment | Setting | U-Net | + WCSP | Δ mIoU [95% CI] / verdict |
|---|---|---|---|---|
| val, 2,124 tiles | handwritten b, λ = 8 | mIoU 0.6489, acc 0.8414 | mIoU **0.6507**, acc 0.8419 | +0.0019 [+0.0010, +0.0027] |
| val, 2,124 tiles | data-derived b, λ = 8 | mIoU 0.6489, acc 0.8414 | mIoU 0.6506, acc **0.8422** | +0.0018 [+0.0011, +0.0024] |
| val, 2,066 tiles | toulbar2 instead of ICM, λ = 4 | mIoU 0.6461 | 0.6464 (ICM: 0.6479) | toulbar2 times out (20 s/block); worse than ICM |
| dev, 185 tiles | + image term f | mIoU 0.7097 (no f) | 0.7087–0.7092 | slightly worse; dropped |

**Why it helps a little on BCSS:**
- Every pixel is a variable, and the unary carries the U-Net's certainty.
- So the rules relabel the *unsure* boundary pixels and stray specks, which is where the U-Net makes edge mistakes.
- Most of the gain is in the "other" class (precision 0.604 → 0.622).

Click a section to expand it.

<details>
<summary><b>How the data-derived b table is computed, and how it differs</b></summary>

#### The b table: handwritten vs. data-derived

Each cell is **handwritten / data-derived**. The table is symmetric, so only the upper triangle is shown; the diagonal is 0.

| | Tumor | Inflam. | Stroma | Necrosis | Other |
|---|---|---|---|---|---|
| **Tumor** | 0 | 2 / 5.2 | 1 / 3.0 | 1 / 4.1 | 9 / 4.5 |
| **Inflam.** | | 0 | 2 / 2.2 | 1 / 6.9 | 8 / 3.2 |
| **Stroma** | | | 0 | 5 / 3.6 | 1 / 2.3 |
| **Necrosis** | | | | 0 | 10 / 4.9 |
| **Other** | | | | | 0 |

- **Handwritten:** the project notes. "Other" is the notes' *healthy*.
- **Data-derived:** how much less often two classes touch in the training ground truth than chance predicts (pointwise mutual information):
  - cost = −log( P(a next to b) / (P(a)·P(b)) ).
  - **Why divide by P(a)·P(b):** so a rare class like necrosis isn't penalized just for being rare.
  - +1 smoothing; negative values are set to 0; rescaled to the handwritten mean (4.0) so λ means the same for both.
- **Key differences:**
  - **Other:** the notes punish anything next to healthy (8–10). BCSS "other" is fat, vessels and similar tissue, which touches everything (data: 2.3–4.9).
  - **Inflammatory–necrosis:** the rarest pair in the data (6.9), but it costs only 1 in the notes.
  - **Triangle inequality:** the handwritten table breaks it. Tumor → stroma → other costs 1 + 1 < 9, so the cheapest "fix" is often a **fake 1 px stroma strip** between tumor and other (see the "hurts" example). The data-derived table almost removes this incentive (3.0 + 2.3 vs. 4.5).

</details>

<details>
<summary><b>Dev sweeps: λ × b table, image term f</b></summary>

#### Step 1. How strong should the rules be, and which b table? (dev, 185 tiles)

U-Net alone: pixel acc 0.8555, mIoU 0.7067.

| λ | Handwritten: acc | mIoU | Δ mIoU [95% CI] | Data-derived: acc | mIoU | Δ mIoU [95% CI] |
|---:|---:|---:|---|---:|---:|---|
| 0.5 | 0.8560 | 0.7082 | +0.0015 [+0.0007, +0.0020] | 0.8562 | 0.7082 | +0.0015 [+0.0008, +0.0021] |
| 1 | 0.8561 | 0.7086 | +0.0019 [+0.0009, +0.0026] | 0.8563 | 0.7085 | +0.0018 [+0.0008, +0.0027] |
| 2 | 0.8563 | 0.7094 | +0.0027 [+0.0014, +0.0035] | 0.8564 | 0.7089 | +0.0022 [+0.0011, +0.0032] |
| 4 | 0.8562 | 0.7096 | +0.0029 [+0.0013, +0.0038] | **0.8565** | 0.7090 | +0.0023 [+0.0012, +0.0033] |
| 8 | 0.8562 | 0.7097 | +0.0030 [+0.0013, +0.0040] | 0.8564 | 0.7090 | +0.0023 [+0.0012, +0.0033] |
| 16 | 0.8562 | **0.7100** | **+0.0032** [+0.0015, +0.0041] | 0.8565 | 0.7090 | +0.0023 [+0.0012, +0.0034] |
| 32 | 0.8561 | 0.7098 | +0.0031 [+0.0013, +0.0042] | 0.8565 | 0.7090 | +0.0023 [+0.0012, +0.0034] |

- **Every setting helps, and every CI is above 0.**
- **The gain levels off by λ ≈ 4–8.** Beyond that, the rules have already fixed every pixel the U-Net is unsure about. Confident pixels never flip, so a larger λ does nothing more.
- **Handwritten wins on mIoU and data-derived on pixel accuracy, by tiny margins inside the CIs.** Effectively a tie.
- **Decision:** use λ = 8 for both tables on val.

#### Step 2. Does the image term f help? (dev, handwritten b)

| λ | β = 0 (no f) | β = 1 | β = 2 | β = 4 |
|---:|---:|---:|---:|---:|
| 4: mIoU (Δ) | 0.7096 (+0.0029) | 0.7095 (+0.0027) | 0.7089 (+0.0022) | 0.7085 (+0.0018) |
| 8: mIoU (Δ) | 0.7097 (+0.0030) | 0.7092 (+0.0025) | 0.7091 (+0.0024) | 0.7087 (+0.0020) |

- **No. It hurts slightly, and more as β grows.**
- **Why:** in H&E images the strongest color edges are nucleus vs. background *inside* one tissue type, not boundaries between tissues. So f protects the wrong edges.
- **Decision:** drop f.

</details>

<details>
<summary><b>Val results, per class, ICM vs. toulbar2, examples</b></summary>

#### Step 3. Does it hold on unseen hospitals? (val, 2,124 tiles, ICM)

U-Net alone: pixel acc **0.8414**, mIoU **0.6489**. This matches the repo's earlier U-Net benchmark (0.649).

| λ | Handwritten: acc | mIoU | Δ mIoU [95% CI] | Data-derived: acc | mIoU | Δ mIoU [95% CI] |
|---:|---:|---:|---|---:|---:|---|
| 0.5 | 0.8418 | 0.6499 | +0.0011 [+0.0006, +0.0014] | 0.8419 | 0.6499 | +0.0010 [+0.0007, +0.0014] |
| 1 | 0.8418 | 0.6503 | +0.0014 [+0.0008, +0.0019] | 0.8421 | 0.6502 | +0.0014 [+0.0009, +0.0018] |
| 2 | 0.8419 | 0.6505 | +0.0017 [+0.0010, +0.0023] | 0.8422 | 0.6506 | +0.0017 [+0.0011, +0.0023] |
| 4 | 0.8419 | 0.6507 | +0.0018 [+0.0010, +0.0026] | 0.8422 | 0.6506 | +0.0018 [+0.0011, +0.0024] |
| **8** | 0.8419 | **0.6507** | **+0.0019** [+0.0010, +0.0027] | **0.8422** | 0.6506 | +0.0018 [+0.0011, +0.0024] |

- **Same picture as dev.** A small, reliable gain that levels off by λ ≈ 4, and the two tables are tied.
- **The gain is smaller than on dev** (+0.0019 vs. +0.0030). New hospitals look different, so the U-Net's mistakes there are more often whole regions, which the WCSP can't fix.
- **About 0.9% of pixels change.** That's +0.05–0.08 pp of pixel accuracy, so the mIoU gain comes from the edges of the smaller classes.

#### Per class (val, λ = 8)

| Method | Tumor | Stroma | Inflam. | Necrosis | Other |
|---|---:|---:|---:|---:|---:|
| **IoU** — U-Net alone | 0.813 | 0.721 | 0.617 | 0.669 | 0.424 |
| WCSP, handwritten b | 0.813 | 0.722 | 0.618 | 0.670 | **0.431** |
| WCSP, data-derived b | 0.814 | 0.722 | 0.618 | 0.671 | 0.427 |
| **Precision** — U-Net alone | 0.892 | 0.828 | 0.800 | 0.805 | 0.604 |
| WCSP, handwritten b | 0.891 | 0.827 | 0.802 | 0.809 | **0.622** |
| WCSP, data-derived b | 0.893 | 0.828 | 0.803 | 0.807 | 0.611 |
| **Recall** — U-Net alone | 0.902 | 0.848 | 0.730 | 0.798 | 0.586 |
| WCSP, handwritten b | 0.903 | 0.850 | 0.729 | 0.795 | 0.584 |
| WCSP, data-derived b | 0.903 | 0.850 | 0.729 | 0.799 | 0.586 |

- **Most of the mIoU gain is the "other" class, through precision** (0.604 → 0.622 with handwritten b).
- **Why:** the handwritten table makes "other" very expensive next to tumor, inflammatory and necrosis (8–10). Stray "other" specks inside those regions get relabeled, and recall barely moves.
- The data-derived table penalizes "other" less, so it cleans fewer specks.

#### Step 4. Does a real WCSP solver do better than greedy ICM? (val, handwritten b, λ = 4)

Same problem and same blocks; only the search inside each block differs. Compared on the 2,066 val tiles both finished (some toulbar2 jobs hit CARC filesystem errors).

| Solver (inside each block) | Total cost vs. U-Net start | Pixel acc | mIoU | Pixels changed | s/tile |
|---|---:|---:|---:|---:|---:|
| none (U-Net alone) | 100% | 0.8423 | 0.6461 | — | — |
| ICM (greedy) | **52.6%** | **0.8428** | **0.6479** | 0.92% | 12 |
| toulbar2 (20 s per block) | 98.6% | 0.8425 | 0.6464 | 0.11% | 86 |

- **Within the time limit, toulbar2 finds much worse solutions than ICM.** It cuts the cost by only 1.4%, against 47% for ICM, and has a higher cost than ICM on 78% of tiles.
- **Why:** a block has 10,000–25,000 variables. toulbar2's exact search (depth-first branch and bound) can't get far in 20 s, so it returns little more than its starting point. ICM's greedy local moves are a better fit at this scale.
- **Lower cost also gave higher mIoU here**, so a better solver could help a bit.
- **To make toulbar2 competitive:**
  - start it from the ICM solution;
  - use its local-search mode (VNS);
  - give it smaller blocks or more time.

#### Examples (val, handwritten b, λ = 4)

Panels: image | ground truth | U-Net | WCSP | changed pixels.

- **Mixed tile:** U-Net 77.9% → WCSP 78.2%. Ragged edges are smoothed and stray specks removed.
  ![mixed](../experiments/bcss_wcsp/figures/val_example_mixed.png)
- **Helps:** U-Net 65.2% → WCSP 66.4%.
  ![helps](../experiments/bcss_wcsp/figures/val_example_helps.png)
- **Hurts:** U-Net 24.2% → WCSP 22.7%.
  - The U-Net calls this tumor "other", and confident regions can't be fixed.
  - The handwritten table also draws 1 px stroma strips along tumor/other edges (see the b table).
  ![hurts](../experiments/bcss_wcsp/figures/val_example_hurts.png)

</details>

---

## 4. Takeaways

- **The WCSP mostly cleans boundaries.**
  - The rules reach at most 2√2 px, and only flip pixels the classifier is unsure about.
  - So they can't fix a whole region or nucleus the classifier got confidently wrong, which is where most errors are.
- **Classifier certainty matters.** With −log p unaries (BCSS) the WCSP gives a small but reliable gain. With a flat "λ per change" unary (PanNuke) its changes are mostly wrong.
- **The handwritten b is a reasonable but rough prior.**
  - On BCSS it ties a data-derived table.
  - It breaks the triangle inequality, which creates fake 1 px stroma strips.
  - On PanNuke its pairwise pressure doesn't match the true classes.
- **Next:**
  - learn b on dev for accuracy;
  - use region- or nucleus-level variables (superpixels, HoVer-Net instances) so the rules can act on whole objects;
  - save HoVer-Net class probabilities so PanNuke can use −log p unaries;
  - for toulbar2, start from the ICM solution and use its local search (VNS).

<details>
<summary><b>Reproduce (BCSS)</b></summary>

```bash
python -m wcsp predict   --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split val            # U-Net once, cached
python -m wcsp solve     --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split val --run NAME --set problem.lam=8
python -m wcsp summarize --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split val --run NAME
python experiments/bcss_wcsp/make_figures.py --id <val tile id>
```

On SLURM, `experiments/bcss_wcsp/slurm/wcsp_multi.sbatch` splits a run into array tasks × 16 CPUs.

</details>
