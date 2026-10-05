# PanNuke Benchmark

Experiments using the PanNuke histopathology dataset.

## Goal

Use PanNuke to test a baseline model for nucleus detection and classification, then examine how the predictions could later be converted into variables for the WCSP solver.

## What to Do

1. **Download the PanNuke dataset**
   - Use the benchmark link provided by the professor.
   - Keep the official dataset folds/splits.

2. **Understand the labels**
   PanNuke contains five main nucleus classes:
   - Neoplastic
   - Inflammatory
   - Connective
   - Dead
   - Non-neoplastic epithelial

3. **Run a baseline**
   - Start with **HoVer-Net**.
   - Use a pretrained PanNuke checkpoint.
   - If time allows, consider another model such as CellViT for comparison.

4. **Record the results**
   Save basic results such as:
   - Precision
   - Recall
   - F1 score
   - Panoptic Quality (PQ), if available
   - Approximate runtime

5. **Inspect the predictions**
   Look at example images and check:
   - Are nuclei detected correctly?
   - Which cell types are commonly confused?
   - Are the predicted cell locations and classes usable for the WCSP pipeline?

6. **Document important decisions**
   Record:
   - Baseline model used
   - Dataset fold/split
   - Pretrained checkpoint used
   - Important preprocessing steps
   - Evaluation metrics
   - Any issues encountered

---

## Tasks

### Aakanksha - Dataset Setup and HoVer-Net Inference

HoVer-Net was selected because it performs both nucleus instance segmentation and classification and has a pretrained PanNuke model. Its nucleus-level outputs also align naturally with the eventual WCSP representation.

Completed:

- Dataset: **PanNuke Fold 2**
- Fold 2 contains **2,523 256x256 histopathology patches**
- Baseline: pretrained `hovernet_fast-pannuke`
- Inference environment: Google Colab GPU
- Verified the PanNuke image, tissue type, and mask structure
- Generated sample input and ground-truth visualizations
- First tested the inference pipeline on 10 sample images
- Successfully ran HoVer-Net on **all 2,523 Fold 2 images**
- Saved nucleus-level predictions including:
  - Predicted cell type
  - Confidence
  - Centroid
  - Bounding box
  - Instance segmentation map
- Generated a sample prediction overlay

Full Fold 2 inference produced:

- **25,301 predicted nuclei**
- Predictions in **2,338 images**
- **185 images with no predicted nuclei**
- Mean prediction confidence: approximately **0.912**

Predicted class counts:

| Predicted Class | Count |
|---|---:|
| Neoplastic | 10,937 |
| Connective | 6,331 |
| Inflammatory | 3,686 |
| Non-neoplastic epithelial | 3,605 |
| Dead | 555 |
| Background | 187 |

Saved outputs include:

- `hovernet_fold2_predictions.csv`
  - One row per predicted nucleus
  - Contains image index, centroid, predicted type, confidence, and bounding box

- `hovernet_fold2_image_summary.csv`
  - One row per Fold 2 image
  - Contains number of predicted nuclei and mean confidence

- `hovernet_instance_maps_*.npz`
  - Predicted instance segmentation maps saved in batches
  - Intended for comparison against PanNuke ground-truth masks

- `PanNuke_HoVerNet_Inference.ipynb`
  - Documents the inference setup and experiment

---
### Abhishek - Evaluation Against Ground Truth

Compared the full Fold 2 HoVer-Net predictions (all 2,523 patches) against the ground truth, which is `masks.npy` from the Fold 2 zip. No new inference was run, this only uses the saved instance maps and prediction CSV.

HoVer-Net fast only outputs the middle 164x164 of each 256x256 patch, so the ground truth is cropped to that same area before comparing. That leaves 28,214 ground-truth nuclei against 25,301 predicted.

| Metric | Value |
| --- | ---: |
| Detection precision | 0.881 |
| Detection recall | 0.790 |
| Detection F1 | 0.833 |
| Class accuracy of detected nuclei | 0.823 |
| Detection + classification F1 | 0.686 |
| Binary PQ | 0.645 |
| Multi-class PQ | 0.437 |

| Class | Ground truth | Predicted | Precision | Recall | F1 | PQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neoplastic | 10,937 | 10,937 | 0.741 | 0.741 | 0.741 | 0.580 |
| Inflammatory | 4,867 | 3,686 | 0.783 | 0.593 | 0.675 | 0.413 |
| Connective | 7,904 | 6,331 | 0.704 | 0.564 | 0.626 | 0.398 |
| Dead | 378 | 555 | 0.292 | 0.429 | 0.347 | 0.382 |
| Non-neoplastic epithelial | 4,128 | 3,605 | 0.761 | 0.664 | 0.709 | 0.392 |

(the neoplastic counts being equal is a coincidence, only 8,103 of them are the same nuclei)

What we see:

- It finds most nuclei. 21% of the real ones are missed and 12% of the predictions don't match any real nucleus.
- The class is the weaker part. 18% of the detected nuclei get the wrong class.
- Dead is the hardest class. 42% of dead nuclei are missed and only 162 of the 555 dead predictions are right. It is also the rarest class.
- Most common mix-ups: connective predicted as neoplastic (715 nuclei), and inflammatory and connective swapped both ways (468 and 464).
- For the WCSP, relabeling could fix wrong classes but it can't bring back nuclei that were never detected.

The full confusion matrix and per-tissue numbers are in `results/evaluation/`.

How it was evaluated:

- Detection: a predicted nucleus matches a ground-truth nucleus if their centroids are within 12 pixels, one-to-one. This is the radius the HoVer-Net paper uses.
- Per-class precision/recall/F1 only count a nucleus if it is both detected and given the right class.
- PQ: masks match if IoU > 0.5. It is averaged per tissue type and then over the 19 tissues, like the PanNuke paper does.
- The 187 predictions that HoVer-Net typed as background count as detections but always as the wrong class.
- The instance maps don't store the class, so it is looked up from the CSV (k-th label in the map = `nucleus_id` k).
- We couldn't find which folds the pretrained checkpoint was trained on. If Fold 2 was one of them these numbers are a bit optimistic.

The confidence values in the CSV are not accuracy and are not used here.

#### Other baseline options

- **CellViT**: same kind of output as HoVer-Net (instance masks + the five PanNuke classes) but with a vision transformer encoder. It has pretrained PanNuke checkpoints and reports better PQ than HoVer-Net, so it is the most natural second baseline. It is a bigger model and heavier to set up and run.
- **StarDist**: predicts each nucleus as a star-convex polygon. It is fast and good at separating touching nuclei, and it has a version that also predicts classes. We don't know of a ready PanNuke checkpoint for it, so we would probably need to train it ourselves.

We went with HoVer-Net because it does segmentation and classification in one model, it is the baseline used in the PanNuke paper, and there is a pretrained PanNuke checkpoint in TIAToolbox that runs on a free Colab GPU in about a minute for the whole fold. Each output is a nucleus with a location and a class, which is what we need for the WCSP variables.

---

### Kashvi - Prediction Inspection and WCSP Mapping

- Dataset: PanNuke Fold 2, first 10 sample 256x256 patches, matching the existing inference run.
- Baseline: pretrained `hovernet_fast-pannuke`; analysis selects patches 000 through 009 from `fold2/hovernet_fold2_predictions.csv` (55 nuclei). No new inference was run for these results.
- Scope: ten-patch exploratory analysis, not a full-dataset benchmark.

**Prediction inspection**

HoVer-Net fast returns a central 164x164 output from each 256x256 input. Shift output centroids by **(+46, +46)** before plotting on the original patch. The dashed box in the [inspection figure](results/sample_predictions/prediction_inspection.png) marks the predicted area. Nuclei outside it are not counted as missed detections.

Among 40 unique ground-truth instances associated with predictions by rounded-centroid containment, four class-disagreement flags were found:

| Patch / CSV nucleus ID | Ground-truth class at centroid | Predicted class |
| --- | --- | --- |
| 002 / 0 | Neoplastic | Connective |
| 003 / 3 | Inflammatory | Connective |
| 008 / 2 | Connective | Neoplastic |
| 009 / 5 | Connective | Neoplastic |

The figure shows patches 002, 003, 008, and 009 with these predictions numbered. Multiple predictions inside one ground-truth instance were excluded as ambiguous. These are qualitative flags: the CSV has no segmentation masks/contours, so this is not one-to-one mask matching or an accuracy/confusion-rate estimate. Instance-level evaluation can validate them. These ten patches contain no dead or non-neoplastic epithelial ground-truth nuclei, so those classes cannot be assessed here.

**WCSP mapping**

- One detected nucleus becomes one variable with the five cell classes as its domain. Keep its patch ID and centroid for spatial relationships.
- With full class probabilities, unary costs can be `U_i(c) = -log(max(p_i(c), 1e-8))`; likely labels have lower cost.
- The CSV's single `confidence` value is a selected-class pixel-vote fraction, not a calibrated five-class probability vector. Full probabilities must be retained before the model's argmax for probability-based unary costs; do not invent the remaining class probabilities.
- A cluster can become one shared-label variable with cost `U_G(c) = sum(U_i(c) for i in G)`. This reduces variables but may erase real mixtures of cell types. Relabeling also cannot recover undetected nuclei.

**Clustering comparison**

Cluster each patch separately using centroids. K-means uses `K=ceil(N/3)`, seed 0, and 10 initializations. DBSCAN uses `eps=40` pixels and `min_samples=2`, retaining each noise nucleus as a separate variable. Spatially constrained agglomerative clustering uses Ward linkage on a 40-pixel radius graph, forming `ceil(component_size/3)` groups within each connected component; disconnected components are never merged. Settings are exploratory.

| Method | Remaining variables | Reduction | Mixed predicted-class groups | Largest group |
| --- | ---: | ---: | ---: | ---: |
| No clustering | 55 | 0% | 0 | 1 |
| K-means | 21 | 61.8% | 9 | 4 |
| DBSCAN | 47 | 14.5% | 0 | 3 |
| Spatial Ward | 47 | 14.5% | 0 | 3 |

**Initial recommendation: DBSCAN.** It matches Ward's summary here with simpler settings and preserves isolated nuclei. K-means provides greater reduction but mixes predicted classes. Methods have different variable budgets; no solver-speed or ground-truth accuracy improvement is established. Predicted-class agreement is not proof of biological homogeneity.

## How to Run

The included [Fold 2 sample](data/sample_fold2/README.md) contains the first ten images and their matching ground-truth masks. Run local commands from the repository root.

### Aakanksha - HoVer-Net Inference

1. Open [PanNuke_HoVerNet_Inference.ipynb](PanNuke_HoVerNet_Inference.ipynb) in Google Colab and select a GPU runtime.
2. Upload `data/sample_fold2/images.npy` to `/content/`. Generate the ten input PNGs before running the notebook's image-list cell:

```python
import numpy as np
from PIL import Image

images = np.load("/content/images.npy")
for i, image in enumerate(images):
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(f"/content/image_{i:03d}.png")
```

3. Run the notebook's setup, model-loading, inference, and CSV-export cells in order.
4. The sample cells export `hovernet_predictions.csv`. The current inspection script uses the checked-in `results/sample_predictions/fold2/hovernet_fold2_predictions.csv` and selects image indices 0 through 9. To regenerate that full-fold file, use the complete Fold 2 inputs and the notebook's full-fold inference/export cells.

### Abhishek - Ground-Truth Evaluation

The evaluation needs the complete Fold 2 ground truth, which is too large for the repository. Download the official Fold 2 archive (about 660 MB) and extract the masks and tissue types into the ignored `data/` directory:

```bash
cd benchmarks/pannuke/data
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip
mkdir -p masks/fold2 images/fold2
unzip -j fold_2.zip '*/masks/fold2/masks.npy' -d masks/fold2
unzip -j fold_2.zip '*/images/fold2/types.npy' -d images/fold2
cd ../../..
```

The extracted `masks.npy` is about 7.4 GB; the script memory-maps it, so it does not need that much RAM. Install the dependencies and run the evaluation:

```bash
python -m pip install numpy pandas scipy
python benchmarks/pannuke/scripts/evaluate_ground_truth.py
```

Use `--masks` and `--types` if the files are stored elsewhere. The script reads the checked-in instance maps and prediction CSV, prints the tables above, and writes them to `results/evaluation/`:

- [Overall metrics](results/evaluation/fold2_overall_metrics.csv)
- [Per-class metrics](results/evaluation/fold2_class_metrics.csv)
- [Confusion matrix](results/evaluation/fold2_confusion_matrix.csv)
- [Per-tissue metrics](results/evaluation/fold2_tissue_metrics.csv)

### Kashvi - Prediction Inspection and Clustering

Install the analysis dependencies in your Python environment:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib
```

Run the analysis:

```bash
python benchmarks/pannuke/scripts/inspect_and_cluster.py
```

The script reads the included sample arrays and saved prediction CSV. It prints the class-disagreement flags and clustering summary, then generates:

- [Clustering results](results/clustering_comparison.csv)
- [Inspection figure](results/sample_predictions/prediction_inspection.png)

---

## Connection to WCSP

Each predicted nucleus contains a location, predicted class, and confidence score. These can be used to construct unary costs and spatial relationships between nearby nuclei.

Because using every predicted nucleus directly would create a large WCSP, a clustering phase will be used to reduce the number of variables. The clustering method should preserve important spatial and biological structure while reducing problem size.
