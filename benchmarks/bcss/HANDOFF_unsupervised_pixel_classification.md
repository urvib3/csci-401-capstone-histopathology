# Handoff: BCSS unsupervised pixel classification

Oct 4, 2026 · @rn-1

## Summary

This task measures how well **unsupervised pixel clustering** can label BCSS tissue (tumor, stroma, inflammatory, necrosis, other) without training a model. Pixels are clustered by stain colour with k-means or a Gaussian mixture, each cluster is named with the majority tissue class of its train pixels, and the result is scored on the held-out-hospital val split.

- **Why it matters:** it is the cheapest possible tissue classifier and the reference point for the project's plan (cluster first, then refine labels with a WCSP). It also tells us whether stain colour alone separates the classes.
- **Status:** code is written, tested and on main, but it has not been run yet. The first full run (12 configurations) is yours; results land in `benchmarks/bcss/results/clustering_semantic.csv`.
- **Your job:** own the results, interpret them for the Deliverable 3 report, and push the method further (see Next steps).

## Method

The pipeline clusters on unlabeled train pixels, uses train labels only to name the clusters, then scores on val.

1. **Stain normalization (optional).** Each tile is mapped onto a reference H&E stain basis with Macenko normalization. The run compares with and without it.
2. **Pixel features (6 per pixel).** Hematoxylin, eosin and DAB optical densities (scikit-image `rgb2hed`, rescaled to about 0–1), plus a Gaussian-smoothed copy of the same three (sigma 4 px) for local context.
3. **Fit clusters, no labels.** 300,000 pixels sampled from 600 random train tiles. Fitted with mini-batch k-means and with a full-covariance Gaussian mixture (GMM), each for k = 5, 8 and 16.
4. **Assign every pixel.** Soft cluster scores (GMM probabilities, or a softmin over k-means distances), each smoothed spatially with sigma 3 px, then the highest score wins. The smoothing removes salt-and-pepper noise.
5. **Name the clusters.** On 800 other random train tiles, count cluster versus ground-truth class. Each cluster takes its majority class. For k = 5 a one-to-one Hungarian matching is also scored.
6. **Score on val.** All 2,768 val tiles: pixel accuracy, mIoU, mean Dice, and IoU/Dice per class, ignoring don't-care pixels.

## Data and splits

The task uses the BCSS\_512 variant of the Kaggle BCSS release: 512x512 tiles with the 22 original label codes, merged to 5 coarse classes by `bcss.coarse_lut()`. Tiles are downsampled to 256x256 for this task.

| Split | Tiles | ROIs | Used for |
| --- | --- | --- | --- |
| train, fit part | 5,307 | 92 | fitting clusters (600 tiles) and naming them (800 tiles) |
| train, dev part | 693 | 14 | held out for tuning; unused in the first run |
| val | 2,768 | 45 | final scores only; hospitals never seen in train |

- **Dev split:** every 8th train ROI, sorted by name (`dev_split` in `common.py`), so dev tiles never share an ROI with fit tiles.
- **Val:** the Kaggle uploader split by hospital (TCGA sites OL, LL, E2, EW, GM, S3), so val measures generalisation to new stain styles.
- **Class balance (labeled pixels):** train 41.7% tumor, 35.4% stroma, 12.5% inflammatory, 7.5% necrosis, 2.8% other. Val is similar (38.0 / 40.0 / 14.2 / 4.8 / 3.0%). 17.7% of train pixels and 32% of val pixels are don't-care.
- **Caches:** `common.load_split` writes `data/cache/bcss512_<split>_256.npz` on first use (about 2 minutes). `data/` is git-ignored.

## How to run

One command runs all 12 configurations (k-means and GMM, k = 5/8/16, with and without Macenko) in about 35–40 minutes on an M2 MacBook with 16 GB.

1. Pull `main` and set up the environment from the repo root:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt scikit-image scikit-learn scipy pandas matplotlib
   ```
2. Download BCSS (about 8 GB, no Kaggle login) into `data/bcss`, or point `$BCSS_ROOT` at an existing copy:

   ```bash
   .venv/bin/python scripts/download_bcss.py
   ```
3. Run the task:

   ```bash
   .venv/bin/python benchmarks/bcss/scripts/cluster_classical.py --part 2 --workers 4
   ```

- **Resuming:** results for each normalization setting are cached in `data/outputs/classical_semantic_macenko{0,1}.pkl`. A rerun skips finished settings. Delete those files after changing the method.
- **Memory:** peak is about 3 GB. Avoid running it beside other heavy jobs on a 16 GB machine; swapping slowed earlier runs about 10x.
- **Workers:** `--workers` sets parallel processes for labeling tiles. Use your core count minus one or two.
- **Part 1** of the same script (`--part 1`) is the separate superpixel benchmark; `--part all` runs both.

## Code map

Everything lives in `benchmarks/bcss/scripts/` and reuses the `bcss/` package at the repo root.

| File | Function | What it does |
| --- | --- | --- |
| `cluster_classical.py` | `part2` | Runs the whole task: loads data, loops over settings and models, names clusters, scores, writes the CSV |
| `cluster_classical.py` | `fit_pixel_models` | Samples train pixels and fits k-means and GMM for each k |
| `cluster_classical.py` | `pixel_features` | 6-dim features per pixel: HED plus smoothed HED |
| `cluster_classical.py` | `soft_assign`, `cluster_map` | Soft cluster scores, spatial smoothing, final cluster per pixel |
| `cluster_classical.py` | `_label_maps` | Labels a chunk of tiles; run in a process pool |
| `common.py` | `load_split`, `dev_split` | Cached 256 px arrays with coarse labels; ROI-based dev split |
| `common.py` | `macenko_normalize`, `hed_features` | Stain normalization and HED optical density |
| `common.py` | `class_report` | Pixel accuracy, mIoU, mDice and per-class IoU/Dice from a confusion matrix |
| `bcss/metrics.py` | `confusion_matrix` | Confusion matrix that skips don't-care pixels (label 255) |

## Settings and design decisions

All settings are first guesses, fixed before seeing val results; none were tuned.

| Setting | Value | Why |
| --- | --- | --- |
| Resolution | 256x256 (512 downsampled 2x) | 4x cheaper; masks are coarse polygons, so little is lost |
| Features | HED + HED smoothed at sigma 4 px | H&E stains separate nuclei-rich tissue from stroma; the smoothed copy adds local context |
| Algorithms | k-means, full-covariance GMM | The two standard clusterers; GMM gives probabilities usable as WCSP unary costs |
| k | 5, 8, 16 | 5 = one per class; more clusters let one class span several stain modes |
| Fit sample | 300,000 pixels from 600 tiles, seed 0 | Enough for 6-dim features; keeps GMM fitting to minutes |
| k-means softmin temperature | 0.01 | Turns distances into soft scores so smoothing works the same as for GMM |
| Spatial smoothing | sigma 3 px on cluster scores | Removes single-pixel noise before taking the top cluster |
| Naming | majority class per cluster (800 tiles, seed 1) | Simple and many-to-one; Hungarian one-to-one added for k = 5 |
| Macenko reference | standard H&E vectors and max concentrations | Values used by the reference Macenko implementation |

## Outputs and metrics

The run writes one CSV, `benchmarks/bcss/results/clustering_semantic.csv`, with one row per configuration (14 rows: 12 majority-named plus 2 Hungarian rows for k = 5).

| Column | Meaning |
| --- | --- |
| `method`, `k`, `macenko`, `mapping` | Which configuration: algorithm, cluster count, normalized or not, majority or Hungarian naming |
| `pixel_acc` | Share of labeled val pixels classified correctly; inflated by the two big classes |
| `mIoU` | Mean over the 5 classes of overlap / union; the main number to compare |
| `mDice` | Mean Dice per class; same ranking as mIoU, easier to compare with papers |
| `IoU_<class>`, `Dice_<class>` | Per-class scores; shows which tissues colour alone can and cannot separate |
| `sec_total` | Wall time to label the 800 naming tiles plus all val tiles |

- **Reading it:** compare `mIoU` across rows. A class with IoU 0 means no cluster was named for it, not that every pixel was wrong.
- **Reference points:** in an early, incomplete run, codebook clusters of DINOv2 patch features scored 0.16–0.17 val mIoU. A 30-step U-Net smoke test reached about 0.40. Expect colour clustering to sit near or below the first.

## Known issues and caveats

- **"Unsupervised" is partial.** Clustering never sees labels, but naming clusters uses 800 labeled train tiles. Say this in the report.
- **Rare classes can vanish.** With majority naming, necrosis (7.5% of train) and other (2.8%) may get no cluster at all, which caps mIoU.
- **Nothing is tuned.** k, smoothing, temperature and the HED rescaling ranges are hand-picked. Tune on the dev split, never on val.
- **HED ranges are approximate.** `hed_features` clips to fixed ranges chosen by eye for H&E tiles; tiles with unusual staining may saturate.
- **Macenko can fall back.** Tiles with fewer than 100 tissue pixels are left unnormalized.
- **Scored at 256 px.** Numbers are not directly comparable with papers that score at full resolution.
- **macOS stray workers.** Killing the script can leave `multiprocessing` worker processes running. Check with `ps aux | grep multiprocessing` and kill them before rerunning.

## Next steps and open questions

- [ ] Run the task, then write up which configuration wins and which classes fail
- [ ] Add a short results section to `benchmarks/bcss/README.md` and the joint Deliverable 3 report
- [ ] Tune k and smoothing sigma on the dev split, then rescore val once
- [ ] Try texture features (local binary patterns, Gabor) beside HED
- [ ] Name clusters with a small classifier instead of majority vote, so rare classes keep a cluster
- [ ] Turn GMM probabilities into WCSP unary costs: pool to the 32x32 `CellGrid` and take `-log p`
- [ ] Compare with the Phikon-v2 codebook classifier (same naming idea, on pathology features) once its results are in

Open questions:

- Is 256 px resolution enough, or should final numbers be scored at 512?
- Should val stay untouched until the final report, with all tuning on dev?

## Context and links

This is one of three BCSS tracks for Deliverable 3. Questions go to @rn-1.

| Track | Status | Where |
| --- | --- | --- |
| Superpixel benchmark (SLIC, Felzenszwalb, watershed and others) | running | `cluster_classical.py --part 1` |
| Phikon-v2 features: clustering, codebook, linear probe | queued | `dino_features.py` |
| DINOv2 evaluation and U-Net baseline | deferred | `dino_features.py`, `train_unet.py` |

- Design docs: [segmentation + WCSP pipeline](https://github.com/urvib3/csci-401-capstone-histopathology/blob/main/docs/design/segmentation-wcsp-pipeline.md) and [algorithm reference](https://github.com/urvib3/csci-401-capstone-histopathology/blob/main/docs/design/segmentation-algorithms.md)
- Dataset: [BCSS on Kaggle](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss); cite Amgad et al., *Bioinformatics* 2019
- A copy of this handoff is in the repo at `benchmarks/bcss/HANDOFF_unsupervised_pixel_classification.md`.
