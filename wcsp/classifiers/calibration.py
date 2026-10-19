# Temperature scaling: one scalar T fitted on dev so that -log p is a fair
# cost (overconfident classifiers otherwise drown out the pairwise rules).

import numpy as np

from ..labels import IGNORE
from ..model.unary import calibrate


def negative_log_likelihood(probs, target, temperature):
    """Mean -log p(true label) over labeled pixels."""
    scaled = calibrate(probs, temperature, axis=1)          # (N, K, P)
    picked = np.take_along_axis(scaled, target[:, None, :], axis=1)[:, 0]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(pairs, pixels_per_sample=2000, seed=0,
                    grid=np.exp(np.linspace(np.log(0.25), np.log(8.0), 41))):
    """
    pairs:
        Iterable of (probs (K, h, w), target (h, w)) with targets in wcsp
        order; IGNORE pixels are skipped.
    pixels_per_sample:
        Labeled pixels drawn at random from each sample (keeps memory flat).

    returns:
        (best temperature, {temperature: nll}).
    """
    rng = np.random.default_rng(seed)
    probs_list, target_list = [], []
    for probs, target in pairs:
        valid = np.flatnonzero(target.ravel() != IGNORE)
        if not len(valid):
            continue
        keep = rng.choice(valid, min(pixels_per_sample, len(valid)), replace=False)
        probs_list.append(probs.reshape(probs.shape[0], -1)[:, keep])
        target_list.append(target.ravel()[keep])
    probs = np.concatenate(probs_list, axis=1).astype(np.float64)[None]
    target = np.concatenate(target_list).astype(np.int64)[None]

    curve = {float(t): negative_log_likelihood(probs, target, t) for t in grid}
    return min(curve, key=curve.get), curve
