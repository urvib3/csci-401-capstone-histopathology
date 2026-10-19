# Evaluating a labeling against a GridProblem.
#
# energy() is the reference cost every solver and test is checked against.
# neighbor_costs() gives, for every pixel and every label, the pairwise cost
# that pixel would pay against a chosen set of fixed neighbors; it is the
# building block of both ICM and conditioning.

import numpy as np

from ..model.neighborhood import pair_slices


def _pair_cost(problem, o, p, labels_p, labels_q):
    cost = problem.table[labels_p, labels_q]
    if problem.change is not None:
        cost = cost + problem.change[o][p] * (labels_p != labels_q)
    return problem.weights[o] * cost


def partial_energy(problem, labels, mask):
    """Unary costs of pixels in mask plus pair costs with both ends in mask (no constant)."""
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    chosen = np.take_along_axis(problem.unary, labels[..., None], axis=-1)[..., 0]
    total = float(chosen[mask].sum())
    for o, offset in enumerate(problem.offsets):
        p, q = pair_slices(problem.shape, offset)
        both = mask[p] & mask[q]
        cost = _pair_cost(problem, o, p, labels[p], labels[q])
        total += float(cost[both].sum())
    return total


def energy(problem, labels):
    """Total cost of a full (H, W) labeling; only active pixels count."""
    return problem.constant + partial_energy(problem, labels, problem.active_mask)


def neighbor_costs(problem, labels, source):
    """
    Pairwise cost each pixel would pay, per label, against neighbors in source.

    returns:
        (H, W, K) array: out[p, k] = sum over neighbors q of p with
        source[q] of the pair cost of (k at p, labels[q] at q).
    """
    labels = np.asarray(labels, dtype=np.int64)
    source = np.asarray(source, dtype=bool)
    k = problem.num_labels
    choices = np.arange(k)
    out = np.zeros(problem.shape + (k,))
    for o, offset in enumerate(problem.offsets):
        p, q = pair_slices(problem.shape, offset)
        change = None if problem.change is None else problem.change[o][p][..., None]
        weight = problem.weights[o]

        # p's cost against its forward neighbor q = p + offset.
        lq = labels[q]
        cost = problem.table[:, lq].transpose(1, 2, 0)          # [.., k] = b[k, l_q]
        if change is not None:
            cost = cost + change * (choices != lq[..., None])
        out[p] += weight * cost * source[q][..., None]

        # q's cost against its backward neighbor p.
        lp = labels[p]
        cost = problem.table[lp, :]                              # [.., k] = b[l_p, k]
        if change is not None:
            cost = cost + change * (choices != lp[..., None])
        out[q] += weight * cost * source[p][..., None]
    return out


def local_costs(problem, labels):
    """unary + pairwise cost against the current labels of all active neighbors."""
    return problem.unary + neighbor_costs(problem, labels, problem.active_mask)
