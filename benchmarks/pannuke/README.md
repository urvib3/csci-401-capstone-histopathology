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
   - Use a pretrained PanNuke checkpoint if available.
   - If time allows, test another model such as CellViT for comparison.

4. **Record the results**
   Save basic results such as:
   - Precision
   - Recall
   - F1 score
   - Panoptic Quality (PQ), if available
   - Approximate runtime

5. **Inspect the predictions**
   Look at a few example images and check:
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

## Tasks

### Aakanksha - HoVer-Net Inference Results

HoVer-Net was selected because it jointly performs nucleus instance segmentation and classification and has a pretrained PanNuke model. Its nucleus-level outputs also align naturally with our eventual WCSP representation

- Dataset: PanNuke Fold 2
- Baseline: pretrained `hovernet_fast-pannuke`
- Inference environment: Google Colab GPU
- Tested on 10 sample 256x256 PanNuke patches
- Total nuclei detected: 55
- Average nuclei per patch: 5.5
- Overall nucleus-weighted mean confidence: ~0.932
- Runtime for 10 patches: ~0.27 seconds
- HoVer-Net outputs included predicted nucleus type, confidence, centroid, bounding box, and contour.

A sample prediction overlay was generated using the predicted nucleus centroids.

**Abhishek**

- Compare HoVer-Net predictions with the PanNuke ground truth.
- Record Precision, Recall, F1, and PQ if available.
- Create a small results table.
- Note which cell classes perform well or poorly.

### Kashvi - Prediction Inspection and WCSP Mapping

- Dataset: PanNuke Fold 2, first 10 sample 256x256 patches, matching the existing inference run.
- Baseline: pretrained `hovernet_fast-pannuke`; analysis uses the saved `hovernet_predictions.csv` (55 nuclei). No new inference was run for these results.
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
4. Download `hovernet_predictions.csv` and place it in `benchmarks/pannuke/results/sample_predictions/`. The current saved CSV can be used directly for inspection and clustering.

### Abhishek - Ground-Truth Evaluation
TODO

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

Review these outputs alongside the WCSP mapping in Kashvi's section above.

## Connection to WCSP

The predicted nuclei can later become WCSP variables. Model probabilities can be used to create unary costs, while spatial relationships between nearby cells can be used for biological constraints.
