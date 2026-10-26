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


def contrast_bootstrap(confs, groups, weights, n_boot=2000, seed=0):
    """
    The same ROI bootstrap for any weighted sum of mIoUs, e.g. the
    difference in WCSP gain between two classifiers:
    {"wcsp_b": 1, "argmax_b": -1, "wcsp_a": -1, "argmax_a": 1}.

    confs:
        {name: (N, K, K)} per-tile confusion matrices, all on the same tiles.
    weights:
        {name: weight}; the statistic is sum(weight * mIoU(name)).

    returns:
        dict like paired_bootstrap, with "delta_mIoU" the observed statistic.
    """
    groups = np.asarray(groups)
    names, index = np.unique(groups, return_inverse=True)
    sums = {}
    for key in weights:
        conf = np.asarray(confs[key])
        sums[key] = np.zeros((len(names),) + conf.shape[1:])
        np.add.at(sums[key], index, conf)

    def statistic(pick):
        return sum(w * _miou(sums[k][pick].sum(0)) for k, w in weights.items())

    rng = np.random.default_rng(seed)
    stats = np.array([statistic(rng.integers(0, len(names), len(names))) for _ in range(n_boot)])
    low, high = np.percentile(stats, [2.5, 97.5])
    return {
        "delta_mIoU": statistic(np.arange(len(names))),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_not_better": float(np.mean(stats <= 0)),
        "groups": len(names),
    }
