# Design: Unsupervised segmentation + WCSP on segment boundaries

Status: **proposed** (not implemented). Builds on the `bcss/` primitives.
Algorithm background: [segmentation-algorithms.md](segmentation-algorithms.md).

## Goal

Label BCSS tissue regions (coarse classes: tumor, stroma, inflammatory,
necrosis, other) by:

1. over-segmenting each tile/ROI with an **unsupervised** clustering method, then
2. solving a **WCSP** whose variables and cost functions live on the resulting
   segments and their boundaries (Toulbar2 via `pytoulbar2`).

Compared to the current lattice formulation (`CellGrid`, one variable per
16x16 cell), segments give fewer, more meaningful variables and let cost
functions use boundary evidence (length, contrast) instead of a uniform grid.

## Pipeline

```
tile / stitched ROI
  -> stain normalization (Macenko)
  -> features (HED channels, texture, or pathology-encoder embeddings)
  -> unsupervised over-segmentation (SLIC baseline)
  -> region adjacency graph (RAG): segments = nodes, shared boundaries = edges
  -> cost tables (unary per segment, binary per boundary)
  -> WCSP (Toulbar2)
  -> per-pixel labels -> bcss.metrics (IoU / Dice)
```

### Stage 1: preprocessing

- Stain-normalize to a fixed reference tile (Macenko; Vahadane as an
  alternative). Required because the `512` val split holds out whole
  hospitals; without it clusters partly encode site rather than tissue.
- Convert to HED with `skimage.color.rgb2hed` for colour features.

### Stage 2: unsupervised over-segmentation

Baseline: **SLIC** on HED channels, `n_segments` chosen so a 512x512 tile
yields ~200-1000 segments. Alternatives to compare: Felzenszwalb, compact
watershed, and Leiden clustering of patch embeddings (UNI / Phikon /
CTransPath) followed by connected components.

Rule: **over-segment**. In the region-level formulations a segment can only
receive one label, so an under-segmentation error is unrecoverable.

### Stage 3: WCSP formulations

#### A. Region labeling on the RAG (primary)

- Variables: one per segment; domain = coarse classes.
- Unary: `-log p(class | segment)` via `unary_costs_from_probabilities` +
  `to_integer_costs`. Probabilities come from a cluster -> class mapping or a
  small classifier on segment features (see "Naming clusters").
- Binary (adjacent segments): label-compatibility cost from
  `bcss.stats.adjacency_counts` -> `negative_log_costs` (or Potts), scaled by
  shared boundary length and down-weighted by boundary colour contrast.
- Graph is planar and irregular; size is ~1/2-1/5 of the 1024-cell lattice.

#### B. Boundary keep/merge (multicut) — stretch

- Variables: one boolean per RAG edge (cut / merge).
- Costs: merge evidence from feature similarity and edge strength.
- Hard constraints: cycle consistency (no cycle with exactly one cut edge);
  added lazily or via triangle constraints on a triangulated RAG.
- Output is an unlabeled partition, so it still needs A afterwards. Natural
  fit for ILP; compare Toulbar2 against an ILP solver.

#### C. Boundary-band refinement

- Run on the existing `CellGrid`. Cells whose 3x3 neighbourhood is a single
  cluster are fixed to that cluster's label.
- Variables only for cells within *k* cells of a cluster boundary; fixed
  neighbours are folded into those cells' unary costs as constants.
- Shrinks the problem drastically and spends search where the clustering is
  uncertain. Fixes boundary placement, which A cannot.

#### D. Hybrid (target pipeline)

A for region labels, then C to refine boundaries between differently
labelled regions.

### Naming clusters

Clusters have no class identity. Map them to classes using the **train**
split only:

- Hungarian matching (`scipy.optimize.linear_sum_assignment`) on the
  cluster x class overlap table when k = C, or majority class per cluster
  when k > C; or
- a small classifier on segment features whose probabilities feed the unaries.

Never fit this mapping on val.

## Evaluation

- Data: `BCSS_512` train/val (split by hospital; no ROI leakage).
- Metrics: per-class IoU / Dice and mIoU from `bcss.metrics`, ignoring
  don't-care pixels.
- Baselines:
  1. unary-only (argmax per segment / per cell),
  2. greedy RAG merging (`skimage.graph.merge_hierarchical`),
  3. existing lattice WCSP on `CellGrid`.
- Also report: number of variables, Toulbar2 solve time, and optimality gap
  when a time limit is hit.

## Implementation plan

1. `bcss/segment.py`: stain normalization, SLIC wrapper, RAG construction
   (adjacency, boundary length, boundary contrast), segment -> pixel upsampling.
2. `bcss/wcsp_region.py`: build formulation A as a `pytoulbar2` model, solve,
   return segment labels.
3. Cluster -> class mapping fit on train; unit tests on synthetic images
   (no dataset needed, like `tests/test_primitives.py`).
4. Evaluation script comparing A against baselines on val.
5. Formulation C on `CellGrid`, then hybrid D.
6. Stretch: formulation B; deep-feature segmentation.

## Open questions

- Target segment count per tile vs. Toulbar2 solve time.
- Learn binary cost weights (boundary length / contrast scaling) or hand-tune?
- Tile-level vs. stitched-ROI-level problems (`stitch_roi`), and how to handle
  segments cut by tile borders.
- Band width *k* for formulation C.
