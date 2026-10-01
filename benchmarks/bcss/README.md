# BCSS Benchmark

Experiments using the BCSS breast cancer semantic segmentation dataset.

## Goal

Use BCSS to test a baseline tissue segmentation model and investigate how large tissue segmentation outputs can be reduced into a smaller representation for the WCSP solver.

## What to Do

1. **Download the BCSS dataset**
   - Use the Kaggle benchmark link provided by the professor.
   - Check the image and segmentation-mask format.

2. **Understand the labels**
   BCSS contains tissue-level annotations such as:
   - Tumor
   - Stroma
   - Inflammatory tissue
   - Necrosis
   - Other tissue

3. **Run a baseline**
   - Start with a **U-Net-style segmentation model**.
   - A pretrained BCSS model can be used if available.
   - The original FCN-based BCSS baseline can also be considered for comparison.

4. **Record the results**
   Save basic metrics such as:
   - Precision
   - Recall
   - F1 score / Dice score
   - IoU
   - Approximate runtime

5. **Inspect the predictions**
   Look at several examples and check:
   - Are tumor and stroma regions separated correctly?
   - Are inflammatory and necrotic regions detected?
   - Which tissue classes are most commonly confused?

6. **Document important decisions**
   Record:
   - Baseline model used
   - Dataset split
   - Pretrained checkpoint or training setup
   - Important preprocessing
   - Metrics
   - Any issues encountered

## Connection to WCSP

BCSS produces tissue-level segmentation rather than individual cell predictions.

Before using the output in the WCSP, the segmentation must be compressed into a smaller number of regions or clusters.

We will compare clustering approaches such as:

- K-means
- DBSCAN
- Spatially constrained agglomerative clustering

The main goal is to reduce the number of WCSP variables while preserving useful spatial and biological information.
