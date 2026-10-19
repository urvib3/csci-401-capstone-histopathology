# Nucleus-level scores for PanNuke.
#
# A pixel labeling becomes a class per detected nucleus by majority vote over
# the nucleus's pixels. Detected and true nuclei are matched one-to-one by
# centroid within 12 px (the HoVer-Net paper's radius); panoptic quality uses
# mask IoU > 0.5. Same rules as benchmarks/pannuke/scripts/evaluate_ground_truth.py,
# so the classifier-only (argmax) row reproduces those numbers' definitions.

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..labels import IGNORE, LABELS, NUM_LABELS

MATCH_RADIUS = 12
MATCH_IOU = 0.5


def instance_classes(labels, instances):
    """Majority label inside each nucleus 1..n; IGNORE where a nucleus has no labeled pixel."""
    instances = np.asarray(instances, dtype=np.int64)
    count = int(instances.max(initial=0))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    inside = (instances > 0) & (labels < NUM_LABELS)
    votes = np.zeros((count + 1, NUM_LABELS), dtype=np.int64)
    np.add.at(votes, (instances[inside], labels[inside]), 1)
    classes = votes[1:].argmax(axis=1)
    classes[votes[1:].sum(axis=1) == 0] = IGNORE
    return classes


def _centroids(instances, count):
    ys, xs = np.nonzero(instances)
    ids = instances[ys, xs]
    area = np.bincount(ids, minlength=count + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.bincount(ids, weights=ys, minlength=count + 1)[1:] / area
        x = np.bincount(ids, weights=xs, minlength=count + 1)[1:] / area
    return np.column_stack([y, x])


def _iou(truth, pred, n_truth, n_pred):
    width = n_pred + 1
    overlap = np.bincount(
        truth.ravel().astype(np.int64) * width + pred.ravel(), minlength=(n_truth + 1) * width
    ).reshape(n_truth + 1, width)
    area_t = overlap.sum(axis=1)[1:, None]
    area_p = overlap.sum(axis=0)[None, 1:]
    inter = overlap[1:, 1:]
    return inter / np.maximum(area_t + area_p - inter, 1)


def nucleus_counts(pred_instances, pred_classes, truth_instances, truth_classes):
    """
    Counts for one patch, to be summed over a dataset and passed to nucleus_report.

    returns dict of arrays:
        confusion (K+1, K+1): matched nuclei, rows truth class, cols predicted
                              class; last row/col = unmatched (missed / spurious)
        pq (K, 4): per class TP, FP, FN, summed IoU (masks of that class only)
    """
    pred_instances = np.asarray(pred_instances, dtype=np.int64)
    truth_instances = np.asarray(truth_instances, dtype=np.int64)
    n_pred, n_truth = len(pred_classes), len(truth_classes)
    k = NUM_LABELS
    confusion = np.zeros((k + 1, k + 1), dtype=np.int64)

    rows = cols = np.empty(0, dtype=int)
    if n_pred and n_truth:
        dist = cdist(_centroids(truth_instances, n_truth), _centroids(pred_instances, n_pred))
        dist = np.nan_to_num(dist, nan=1e6)
        rows, cols = linear_sum_assignment(np.where(dist <= MATCH_RADIUS, dist, 1e6))
        keep = dist[rows, cols] <= MATCH_RADIUS
        rows, cols = rows[keep], cols[keep]

    def bucket(c):
        return c if c < k else k

    for r, c in zip(rows, cols):
        confusion[bucket(truth_classes[r]), bucket(pred_classes[c])] += 1
    for r in np.setdiff1d(np.arange(n_truth), rows):
        confusion[bucket(truth_classes[r]), k] += 1
    for c in np.setdiff1d(np.arange(n_pred), cols):
        confusion[k, bucket(pred_classes[c])] += 1

    pq = np.zeros((k, 4))
    iou = _iou(truth_instances, pred_instances, n_truth, n_pred) if n_pred and n_truth else None
    for cls in range(k):
        t_sel = np.asarray(truth_classes) == cls
        p_sel = np.asarray(pred_classes) == cls
        if iou is not None:
            matched = iou[np.ix_(t_sel, p_sel)]
            matched = matched[matched > MATCH_IOU]
        else:
            matched = np.empty(0)
        tp = len(matched)
        pq[cls] = [tp, p_sel.sum() - tp, t_sel.sum() - tp, matched.sum()]
    return {"confusion": confusion, "pq": pq}


def nucleus_report(confusion, pq):
    """Scores from summed nucleus_counts."""
    k = NUM_LABELS
    matched = confusion[:k, :k].sum()
    report = {
        "detection_f1": float(2 * matched / (confusion[:k].sum() + confusion[:, :k].sum()))
        if confusion.sum() else float("nan"),
        "class_acc": float(np.trace(confusion[:k, :k]) / matched) if matched else float("nan"),
    }
    pqs = []
    for cls, name in enumerate(LABELS):
        tp = confusion[cls, cls]
        truth = confusion[cls].sum()
        predicted = confusion[:, cls].sum()
        report[f"F1_{name}"] = float(2 * tp / (truth + predicted)) if truth + predicted else float("nan")
        t, fp, fn, iou_sum = pq[cls]
        value = (iou_sum / t) * (t / (t + 0.5 * fp + 0.5 * fn)) if t else 0.0
        report[f"PQ_{name}"] = float(value)
        pqs.append(value)
    report["mPQ"] = float(np.mean(pqs))
    report["class_f1"] = float(np.nanmean([report[f"F1_{n}"] for n in LABELS]))
    return report
