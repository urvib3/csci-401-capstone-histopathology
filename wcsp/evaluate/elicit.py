# Eliciting the b_ij table from ground truth ("costs can also be elicited
# with statistical analysis", notes p. 4).
#
# For every neighbor type, count how often labels i and j sit at that
# distance in ground-truth masks. The cost of a pair is its negative
# pointwise mutual information, clipped at 0:
#
#   b_ij = max(0, -log( P(i, j) / (P(i) P(j)) ))
#
# so pairs that co-occur less often than chance cost more, and pairs that
# co-occur at least as often as chance (including i == j) are free.

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..labels import NUM_LABELS
from ..model.neighborhood import ALL_TYPES, forward_offsets
from ..model.tables import HANDWRITTEN
from .rules import pair_counts


def fill_background(labels, valid):
    """Give every invalid pixel the label of the nearest valid pixel (PanNuke "cell territories")."""
    labels = np.asarray(labels)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return labels.copy()
    _, (rows, cols) = distance_transform_edt(~valid, return_indices=True)
    return labels[rows, cols]


def counts_by_type(masks, valid_fn=None):
    """
    masks:
        Iterable of (H, W) wcsp label masks (IGNORE skipped).

    returns:
        {neighbor type: (K, K) pair counts}.
    """
    totals = {t: np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64) for t in ALL_TYPES}
    offsets = {t: forward_offsets((t,)) for t in ALL_TYPES}
    for mask in masks:
        for t in ALL_TYPES:
            totals[t] += pair_counts(mask, offsets=offsets[t])
    return totals


def pmi_table(counts, smoothing=1.0, match_scale=True):
    """
    -PMI costs from pair counts (any neighbor type, or their sum).

    match_scale:
        Rescale so the mean off-diagonal cost equals the handwritten table's,
        keeping lam comparable between the two tables.
    """
    counts = np.asarray(counts, dtype=np.float64) + smoothing
    joint = counts / counts.sum()
    marginal = joint.sum(axis=1)
    cost = np.maximum(0.0, -np.log(joint / np.outer(marginal, marginal)))
    np.fill_diagonal(cost, 0.0)
    cost = (cost + cost.T) / 2
    off = ~np.eye(NUM_LABELS, dtype=bool)
    if match_scale and cost[off].mean() > 0:
        cost *= HANDWRITTEN[off].mean() / cost[off].mean()
    return cost


def unlike_rate_by_type(counts):
    """Share of neighboring pairs with different labels, per neighbor type (how fast labels decorrelate)."""
    rates = {}
    for t, c in counts.items():
        total = np.triu(c).sum()
        rates[t] = float((total - np.trace(c)) / total) if total else float("nan")
    return rates
