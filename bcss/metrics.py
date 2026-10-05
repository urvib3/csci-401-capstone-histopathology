# Segmentation metrics for comparing a labeling against a BCSS mask.
#
# Pixels whose ground truth is ignore_index (outside the ROI or excluded)
# are never scored, following the BCSS "don't care" convention.

import numpy as np

from .classes import IGNORE_INDEX


def confusion_matrix(pred, target, num_classes, ignore_index=IGNORE_INDEX):
    """
    returns:
        (num_classes, num_classes) int array; rows are ground truth,
        columns are predictions.
    """
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    valid = target != ignore_index
    index = target[valid].astype(np.int64) * num_classes + pred[valid]
    return np.bincount(index, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def pixel_accuracy(cm):
    total = cm.sum()
    return float(np.trace(cm) / total) if total else float("nan")


def iou_per_class(cm):
    """IoU for each class; NaN for classes absent from both pred and target."""
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def dice_per_class(cm):
    """Dice coefficient for each class; NaN where undefined."""
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def mean_iou(cm):
    return float(np.nanmean(iou_per_class(cm)))
