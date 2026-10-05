# Label statistics over BCSS masks.
#
# These are the empirical quantities a WCSP formulation is likely to need:
# class priors (unary costs without an image model) and how often two
# classes sit next to each other (binary compatibility costs on lattice
# edges).

import numpy as np

from .classes import IGNORE_INDEX


def class_counts(masks, num_classes, ignore_index=IGNORE_INDEX):
    """
    Total pixel count of each class over an iterable of (H, W) masks.

    returns:
        int array of length num_classes.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for mask in masks:
        mask = np.asarray(mask).ravel()
        counts += np.bincount(
            mask[mask != ignore_index], minlength=num_classes
        )[:num_classes]
    return counts


def adjacency_counts(label_grid, num_classes, connectivity=4,
                     ignore_index=IGNORE_INDEX):
    """
    Count how often each ordered pair of classes occurs on neighboring sites.

    label_grid:
        (rows, cols) labels; either a pixel mask or CellGrid.majority_labels.

    returns:
        symmetric (num_classes, num_classes) int array. counts[a, b] is the
        number of neighboring site pairs labeled {a, b}; the diagonal holds
        same-class pairs.
    """
    grid = np.asarray(label_grid)
    shifts = [(grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])]
    if connectivity == 8:
        shifts.append((grid[:-1, :-1], grid[1:, 1:]))
        shifts.append((grid[:-1, 1:], grid[1:, :-1]))
    elif connectivity != 4:
        raise ValueError("connectivity must be 4 or 8.")

    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for a, b in shifts:
        a, b = a.ravel(), b.ravel()
        valid = (a != ignore_index) & (b != ignore_index)
        np.add.at(counts, (a[valid], b[valid]), 1)
    return counts + counts.T - np.diag(np.diag(counts))


def negative_log_costs(counts, smoothing=1.0):
    """
    Turn co-occurrence or class counts into -log probability costs.

    Additive smoothing keeps unseen combinations finite (soft rather than
    hard constraints). Use a large finite value instead if a combination
    should be forbidden.
    """
    counts = np.asarray(counts, dtype=np.float64) + smoothing
    return -np.log(counts / counts.sum())
