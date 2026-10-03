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

**Aakanksha**

- Download one PanNuke fold.
- Set up HoVer-Net with the pretrained PanNuke checkpoint.
- Run inference on a small set of PanNuke images.
- Save prediction outputs and record setup/runtime information.

**Abhishek**

- Compare HoVer-Net predictions with the PanNuke ground truth.
- Record Precision, Recall, F1, and PQ if available.
- Create a small results table.
- Note which cell classes perform well or poorly.

**Kashvi**

- Inspect example predictions and common classification errors.
- Document how nucleus predictions can map to WCSP variables and unary costs.
- Investigate clustering methods for reducing the number of WCSP variables.
- Compare K-means, DBSCAN, and spatially constrained agglomerative clustering.

## Connection to WCSP

The predicted nuclei can later become WCSP variables. Model probabilities can be used to create unary costs, while spatial relationships between nearby cells can be used for biological constraints.
