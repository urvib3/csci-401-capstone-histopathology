# Unary costs: each pixel's fidelity to the pixel classifier.

import numpy as np


def calibrate(probs, temperature=1.0, axis=0):
    """
    Temperature-scale class probabilities: p^(1/T), renormalized.

    Equivalent to dividing the classifier's logits by T. T > 1 softens
    overconfident predictions; T = 1 leaves them unchanged.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    probs = np.asarray(probs, dtype=np.float64)
    if temperature == 1.0:
        return probs
    log = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    log -= log.max(axis=axis, keepdims=True)
    scaled = np.exp(log)
    return scaled / scaled.sum(axis=axis, keepdims=True)


def unary_costs(probs, temperature=1.0, informative=None, eps=1e-6):
    """
    -log p unary costs.

    probs:
        (K, H, W) class probabilities in wcsp label order.
    informative:
        Optional (H, W) bool mask. Pixels where it is False (e.g. PanNuke
        background, where the classifier says nothing about cell type) get
        zero cost for every label, so the pairwise rules decide them.

    returns:
        (H, W, K) float64 costs, shifted so each pixel's best label costs 0.
    """
    probs = calibrate(probs, temperature, axis=0)
    costs = -np.log(np.clip(np.moveaxis(probs, 0, -1), eps, 1.0))
    costs -= costs.min(axis=-1, keepdims=True)
    if informative is not None:
        costs[~np.asarray(informative, dtype=bool)] = 0.0
    return costs
