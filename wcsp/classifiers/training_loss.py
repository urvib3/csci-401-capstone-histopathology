"""Differentiable expected WCSP pairwise cost; probabilities are BCSS ordered."""

import numpy as np
import torch

from ..labels import BCSS_TO_WCSP, IGNORE
from ..model.neighborhood import distance_weights, forward_offsets, pair_slices
from ..model.tables import load_table


def expected_pairwise_cost(probs, target, table="handwritten", eta=1.0,
                           types=(1, 2, 3, 4, 5), normalize=True):
    """Sum w_pq P_p.T b P_q / number of labeled pixels.

    Each unordered local pair appears once, matching the WCSP energy per
    pixel. Pairs touching ignored targets contribute no loss. Both endpoints
    receive gradients; no argmax or ground-truth pair labels are used.
    """
    b = load_table(table)[np.ix_(BCSS_TO_WCSP, BCSS_TO_WCSP)]
    b = torch.as_tensor(b, dtype=probs.dtype, device=probs.device)
    offsets = forward_offsets(types)
    weights = distance_weights(offsets, eta, normalize)
    valid = target != IGNORE
    total = probs.sum() * 0.0
    for offset, weight in zip(offsets, weights):
        p, q = pair_slices(target.shape[-2:], offset)
        pp, pq = probs[(slice(None), slice(None)) + p], probs[(slice(None), slice(None)) + q]
        cost = torch.einsum("bihw,ij,bjhw->bhw", pp, b, pq)
        both = valid[(slice(None),) + p] & valid[(slice(None),) + q]
        total = total + float(weight) * (cost * both).sum()
    return total / valid.sum().clamp_min(1)
