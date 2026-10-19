# Pixel-level scores (BCSS, and PanNuke nucleus pixels).

import numpy as np

from bcss.metrics import confusion_matrix, dice_per_class, iou_per_class

from ..labels import IGNORE, LABELS, NUM_LABELS


def pixel_confusion(labels, target):
    """(K, K) counts, rows = truth, columns = prediction; IGNORE pixels skipped."""
    return confusion_matrix(labels, target, NUM_LABELS, ignore_index=IGNORE)


def pixel_report(confusion):
    """Pixel accuracy, mIoU, mDice and per-class IoU / Dice from a summed confusion matrix."""
    confusion = np.asarray(confusion)
    total = confusion.sum()
    iou, dice = iou_per_class(confusion), dice_per_class(confusion)
    report = {
        "pixel_acc": float(np.trace(confusion) / total) if total else float("nan"),
        "mIoU": float(np.nanmean(iou)),
        "mDice": float(np.nanmean(dice)),
    }
    for name, i, d in zip(LABELS, iou, dice):
        report[f"IoU_{name}"] = float(i)
        report[f"Dice_{name}"] = float(d)
    return report
