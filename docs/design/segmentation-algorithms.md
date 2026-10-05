# Reference: unsupervised segmentation algorithms

Background for [segmentation-wcsp-pipeline.md](segmentation-wcsp-pipeline.md).

## Superpixel / over-segmentation

### SLIC (Simple Linear Iterative Clustering)
Localized k-means in joint (colour, x, y) space.
1. Seed K centres on a grid with spacing S = sqrt(N/K); move each to the
   lowest-gradient pixel in its 3x3 neighbourhood.
2. For each centre, search only a 2S x 2S window; assign pixels by
   `D = sqrt(d_color^2 + (m/S)^2 * d_xy^2)`.
3. Recompute centres as means; repeat ~10 iterations; enforce connectivity.

Knobs: `n_segments` (K), `compactness` (m: high = grid-like, low = edge-hugging).
O(N) per iteration. Predictable size and count -> predictable WCSP size.
Weak on low-contrast stroma edges; run on HED instead of RGB.

### Felzenszwalb–Huttenlocher
Greedy merging on a pixel graph (edge weight = colour difference). Process
edges in increasing weight; merge components A, B if
`w <= min(Int(A) + k/|A|, Int(B) + k/|B|)`, where `Int(C)` is the largest
MST edge inside C.

Knobs: `scale` (k), `sigma`, `min_size`. O(E log E). Adapts segment size to
content (large in homogeneous stroma, small in tumour); sizes vary widely.

### Watershed
Treat an image (gradient, or inverted hematoxylin) as terrain and flood from
markers; basin meeting lines are boundaries.

Knobs: markers (essential; none = heavy over-segmentation), `compactness`
(compact watershed ~ SLIC hybrid). ~O(N log N). Very precise boundaries;
standard for splitting touching nuclei.

### Quickshift
Estimate a Parzen density per pixel in (colour, x, y); link each pixel to its
nearest higher-density neighbour within `max_dist`; each tree is a segment.

Knobs: `kernel_size`, `max_dist`, `ratio`. No K needed. Slow on large ROIs.

### Mean shift
Each point iteratively moves to the kernel-weighted mean of its neighbourhood
until it reaches a density mode; points sharing a mode form a cluster.
Knob: bandwidth. Principled but slow; use on downsampled images.

## Feature clustering (not spatially aware)

Output must be split into connected components to become segments.

### k-means
Initialise k centroids (k-means++), assign points to the nearest, recompute
means, repeat. Features: HED intensities, texture (LBP, Gabor), embeddings.
Assumes round, similar-size clusters; k chosen by elbow / silhouette.

### Gaussian mixture model (GMM)
Soft k-means with per-cluster mean and covariance, fit by EM
(E: responsibilities; M: refit parameters). Handles elongated clusters and
yields `p(cluster | pixel)`, which maps directly to WCSP unaries (`-log p`).
k chosen by BIC / AIC.

### Leiden (vs. Louvain)
Community detection on a kNN graph of embeddings, maximising modularity:
local node moves -> refinement (guarantees connected communities; Louvain
lacks this) -> aggregate and repeat. Knob: `resolution`. Standard for
clustering foundation-model embeddings; finds rare tissue types.

### Deep features (UNI, CONCH, Phikon, CTransPath)
Not clustering algorithms: self-supervised pathology encoders that map a
patch (e.g. 224x224) to a 768-1536-d embedding, which is then clustered.
Most semantic clusters, but patch-level resolution -> blocky boundaries
(motivates boundary-band refinement, formulation C).

## Region adjacency graph (RAG) methods

- **Greedy hierarchical merging** (`skimage.graph.rag_mean_color`,
  `merge_hierarchical`): repeatedly merge the most similar adjacent pair until
  a threshold. Heuristic baseline for the WCSP.
- **Normalized cut** (`cut_normalized`): recursively bipartition minimising
  `cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)`, approximated by the second
  eigenvector of the graph Laplacian (spectral clustering).

## Supporting steps

- **Macenko stain normalization**: RGB -> optical density `-log(I/I0)`, drop
  background, SVD to the top-2 plane, take robust extreme angles as H and E
  stain vectors, re-express pixels in a reference image's stain basis.
  Vahadane uses sparse NMF instead (more robust, slower).
- **Hungarian matching**: build the cluster x class overlap table on train and
  solve the linear assignment problem (`scipy.optimize.linear_sum_assignment`,
  O(n^3)); use majority class per cluster when k > C.

## Comparison

| Method | Choose K? | Spatial | Boundaries | Speed | Role here |
| --- | --- | --- | --- | --- | --- |
| SLIC | yes | yes | good | fast | default over-segmentation |
| Felzenszwalb | scale | yes | good, irregular | very fast | adaptive sizes |
| Watershed | markers | yes | excellent | fast | nuclei / sharp edges |
| Quickshift / mean shift | no | yes | good | slow | unknown K |
| k-means / GMM | yes | no | pixel-wise | fast | colour/texture; GMM gives probs |
| Leiden on deep features | resolution | no | blocky | moderate | semantic clusters |
| RAG merge / Ncut | threshold | yes | inherits base | fast | baseline vs. WCSP |
