# Is the WCSP better than the classifier alone, beyond chance?
#
# Tiles from one ROI are strongly correlated, so the bootstrap resamples
# whole ROIs: draw ROIs with replacement, sum both methods' confusion
# matrices over the drawn tiles, and record the mIoU difference.

import numpy as np

from bcss.metrics import iou_per_class


def _miou(confusion):
    return float(np.nanmean(iou_per_class(confusion)))


def paired_bootstrap(conf_a, conf_b, groups, n_boot=2000, seed=0):
    """
    conf_a, conf_b:
        (N, K, K) per-tile confusion matrices of two methods on the same tiles.
    groups:
        (N,) group id per tile (e.g. ROI); groups are resampled as units.

    returns:
        dict with the observed mIoU difference (b - a), its 95% interval and
        the share of resamples where b is not better.
    """
    conf_a, conf_b = np.asarray(conf_a), np.asarray(conf_b)
    groups = np.asarray(groups)
    names, index = np.unique(groups, return_inverse=True)
    sum_a = np.zeros((len(names),) + conf_a.shape[1:])
    sum_b = np.zeros_like(sum_a)
    np.add.at(sum_a, index, conf_a)
    np.add.at(sum_b, index, conf_b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(names), len(names))
        diffs[i] = _miou(sum_b[pick].sum(0)) - _miou(sum_a[pick].sum(0))
    observed = _miou(conf_b.sum(0)) - _miou(conf_a.sum(0))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta_mIoU": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_not_better": float(np.mean(diffs <= 0)),
        "groups": len(names),
    }
