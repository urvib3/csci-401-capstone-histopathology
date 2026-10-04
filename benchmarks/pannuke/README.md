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

**Goal:** Quantitatively evaluate the full Fold 2 HoVer-Net predictions against PanNuke ground truth.

Use:

- `hovernet_instance_maps_*.npz`
- PanNuke Fold 2 `masks.npy`
- `hovernet_fold2_predictions.csv`

Tasks:

1. Match each predicted instance map with the corresponding Fold 2 ground-truth mask.
2. Evaluate detection and classification performance.
3. Record metrics such as:
   - Precision
   - Recall
   - F1 score
   - Panoptic Quality (PQ), if available
4. Evaluate performance across the five nucleus classes.
5. Identify classes that are commonly confused or difficult to detect.
6. Create a small results table for the final deliverable.
7. Document any evaluation assumptions or preprocessing decisions.

The confidence values in the prediction CSV are model confidence scores and should not be treated as accuracy. Final performance should be measured against the PanNuke ground truth.

---

### Kashvi - Prediction Analysis, Clustering, and WCSP Connection

**Goal:** Analyze the HoVer-Net predictions and investigate how they can be compressed into a smaller WCSP representation.

Use:

- `hovernet_fold2_predictions.csv`
- `hovernet_fold2_image_summary.csv`
- Sample prediction overlays

The prediction CSV provides the main information needed for clustering:

- `centroid_x`
- `centroid_y`
- `predicted_type`
- `predicted_type_id`
- `confidence`
- Bounding box coordinates

Tasks:

1. Inspect the distribution of predicted cell types and confidence scores.
2. Examine example predictions and identify possible classification issues.
3. Investigate clustering approaches for reducing the number of WCSP variables.
4. Compare methods such as:
   - K-means
   - DBSCAN
   - Spatially constrained agglomerative clustering
5. Compare different levels of compression, such as different numbers of clusters.
6. Consider whether clustering preserves:
   - Cell-type information
   - Spatial relationships
   - Biologically meaningful local structure
7. Document how the prediction outputs can map into the WCSP:
   - Nucleus or cluster → WCSP variable
   - Predicted class/confidence → unary cost
   - Centroid/location → spatial constraints
   - Neighbor relationships → biological constraints

The goal is to reduce the number of variables sent to the WCSP solver without losing too much useful spatial or classification information.

---

## Connection to WCSP

HoVer-Net produces nucleus-level predictions that can be used as the starting point for the WCSP formulation.

Each predicted nucleus contains a location, predicted class, and confidence score. These can be used to construct unary costs and spatial relationships between nearby nuclei.

Because using every predicted nucleus directly would create a large WCSP, a clustering phase will be used to reduce the number of variables. The clustering method should preserve important spatial and biological structure while reducing problem size.

The overall pipeline is:

**PanNuke image → HoVer-Net predictions → clustering → WCSP variables and constraints → Toulbar2**
