# WCSP on clusters: group pixels that look like the same tissue, give each group one label.
#
# Pixels are grouped into superpixels (SLIC) on the image colour (Lab) and, optionally,
# the U-Net probabilities. Each cluster c is one variable. Its costs are the pixel
# WCSP's, summed over the cluster:
#
#   unary_c(k)        = sum_{p in c} U_p(k)
#   pair_{c,c'}(k,k') = sum_{(p,q) neighbors, p in c, q in c'} w_pq * b[k, k']   (+ contrast term)
#
# Pixel pairs inside one cluster always agree and cost nothing. So the cluster energy
# of a labeling equals the pixel energy of the piecewise-constant labeling it induces:
# the same objective, restricted to "one label per cluster". The cluster graph is
# small (tens to ~1,000 nodes) and is solved with ICM.
#
# refine: optionally run a pixel solver afterwards, starting from the cluster labels,
# so borders inside clusters can still move.

import time

import numpy as np

from .base import SolveResult
from ..model.neighborhood import pair_slices
from .energy import energy


def superpixels(image, probs, n_segments, compactness, features, prob_weight):
    from skimage.color import rgb2lab
    from skimage.segmentation import slic

    lab = rgb2lab(image[..., :3])
    channels = [lab]
    if features == "both":
        channels.append(prob_weight * probs)
    elif features == "probs":
        channels = [prob_weight * probs]
    stack = np.concatenate(channels, axis=-1)
    # skimage rescales float input to [0, 1] before clustering; scale compactness the same
    # way so it stays in Lab units (otherwise the spatial term dominates and SLIC returns a grid).
    span = float(stack.max() - stack.min())
    if span > 0:
        compactness = compactness / span
    return slic(stack, n_segments=n_segments, compactness=compactness, channel_axis=-1,
                convert2lab=False, start_label=0, enforce_connectivity=True)


def cluster_graph(problem, segments):
    """Cluster unaries (C, K), table weights A (C, C) and contrast weights D (C, C)."""
    if problem.active is not None or (problem.long_edges is not None and len(problem.long_edges)):
        raise ValueError("cluster WCSP needs a fully active grid without long edges.")
    k = problem.num_labels
    n = int(segments.max()) + 1
    flat = segments.ravel()
    unary = np.zeros((n, k))
    np.add.at(unary, flat, problem.unary.reshape(-1, k))
    table_w = np.zeros((n, n))
    change_w = np.zeros((n, n))
    for o, offset in enumerate(problem.offsets):
        p, q = pair_slices(problem.shape, offset)
        a, b = segments[p].ravel(), segments[q].ravel()
        cross = a != b
        a, b = a[cross], b[cross]
        np.add.at(table_w, (a, b), problem.weights[o])
        if problem.change is not None:
            np.add.at(change_w, (a, b), problem.weights[o] * problem.change[o][p].ravel()[cross])
    table_w = table_w + table_w.T          # symmetric: each crossing pair once in each row
    change_w = change_w + change_w.T
    return unary, table_w, change_w


def cluster_icm(unary, table_w, change_w, table, init, max_sweeps=100):
    z = np.array(init, dtype=np.int64)
    k = unary.shape[1]
    choices = np.arange(k)
    neighbors = [np.flatnonzero(table_w[c] + change_w[c]) for c in range(len(z))]
    for _ in range(max_sweeps):
        moved = False
        for c, nb in enumerate(neighbors):
            cost = unary[c].copy()
            if len(nb):
                cost += table_w[c, nb] @ table[:, z[nb]].T
                cost += change_w[c, nb] @ (choices[None, :] != z[nb][:, None])
            best = int(cost.argmin())
            if cost[best] < cost[z[c]] - 1e-12:
                z[c] = best
                moved = True
        if not moved:
            break
    return z


class ClusterSolver:
    """
    n_segments, compactness:
        SLIC settings (roughly how many clusters; higher compactness = rounder).
    features:
        "image" (Lab colour), "probs" (U-Net probabilities) or "both".
    prob_weight:
        Scale of the probability channels relative to Lab (0-100).
    refine:
        Optional pixel solver run afterwards from the cluster labels.
    """

    def __init__(self, n_segments=256, compactness=10.0, features="both", prob_weight=30.0,
                 refine=None, max_sweeps=100):
        self.n_segments = n_segments
        self.compactness = compactness
        self.features = features
        self.prob_weight = prob_weight
        self.refine = refine
        self.max_sweeps = max_sweeps
        self.image = None

    def prepare(self, sample, prediction):
        self.image = sample.grid_image

    def segments(self, problem):
        from .hierarchy import pixel_probs

        probs = pixel_probs(problem)
        image = self.image
        if image is None or image.shape[:2] != problem.shape:
            if self.features != "probs":
                raise ValueError("cluster WCSP needs the tile image (call prepare) for image features.")
            image = np.zeros(problem.shape + (3,), dtype=np.uint8)
        return superpixels(image, probs, self.n_segments, self.compactness, self.features, self.prob_weight)

    def solve(self, problem, init):
        start = time.perf_counter()
        segments = self.segments(problem)
        unary, table_w, change_w = cluster_graph(problem, segments)
        z = cluster_icm(unary, table_w, change_w, problem.table, unary.argmin(1), self.max_sweeps)
        labels = z[segments]
        trace = [{"stage": "clusters", "energy": energy(problem, labels), "clusters": len(z)}]
        if self.refine is not None:
            labels = self.refine.solve(problem, labels).labels
            trace.append({"stage": "refine", "energy": energy(problem, labels)})
        result = SolveResult(labels, energy(problem, labels), time.perf_counter() - start, trace=trace)
        result.info = {"clusters": len(z)}
        return result
