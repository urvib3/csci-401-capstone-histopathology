# Fixing part of an image and keeping only the rest as variables.
#
# This is how the block scheme works: to solve a region with everything else
# held at its current labels, each region pixel's pair costs against fixed
# neighbors become part of its unary cost, the cost of the fixed part becomes
# a constant, and the problem is cropped to the region's bounding box. The
# result is an ordinary GridProblem, so any solver can be used on it, and
#
#   energy(sub, labels[window]) == energy(problem, labels)
#
# for every labeling that agrees with the fixed pixels.

from dataclasses import replace

import numpy as np

from .energy import neighbor_costs, partial_energy


def bounding_box(mask):
    """Smallest (row slice, col slice) containing every True pixel."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        raise ValueError("region is empty.")
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def condition(problem, labels, region):
    """
    Fix every active pixel outside region to its label in labels.

    region:
        (H, W) bool, a subset of the problem's active pixels.

    returns:
        (sub, window): the smaller GridProblem and the (row, col) slices of
        the original image it covers. sub.active marks the region pixels.
    """
    region = np.asarray(region, dtype=bool)
    active = problem.active_mask
    if (region & ~active).any():
        raise ValueError("region contains pixels that are already fixed.")
    fixed = active & ~region

    unary = problem.unary + neighbor_costs(problem, labels, fixed)
    constant = problem.constant + partial_energy(problem, labels, fixed)

    window = bounding_box(region)
    sub = replace(
        problem,
        unary=unary[window],
        change=None if problem.change is None else problem.change[(slice(None),) + window],
        active=region[window],
        constant=constant,
    )
    return sub, window
