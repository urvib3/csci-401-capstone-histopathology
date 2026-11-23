# Region rules applied after the inner solver. Each relabels whole 8-connected regions of
# a class to the other class with the lowest summed unary over the region (the U-Net's
# best alternative for those pixels), so the result stays close to the classifier.
#
#   min_area   {class: px}      regions smaller than this are removed. Fit ground truth:
#                               necrosis regions have median 26,575 px and only 0.14% of
#                               necrosis pixels lie in regions under 1,024 px, while the
#                               U-Net's median necrosis region is 31 px.
#   border     [{cls, needs, min_share}] or {name: {cls, needs, min_share}}
#                               regions of cls whose outside border is less than min_share
#                               class `needs` are removed, e.g. inflammatory infiltrates sit
#                               in stroma (96% of fit regions mostly border stroma).
#                               Regions without a border (filling the tile) are kept.
#   min_share  {class: fraction}
#                               a class covering less than this share of the tile is removed
#                               entirely (present classes are rarely tiny).

import time

import numpy as np
from scipy import ndimage

from .base import SolveResult
from .energy import energy

EIGHT = np.ones((3, 3))


def _relabel(out, region, unary, cls):
    cost = unary[region].sum(0)
    cost[cls] = np.inf
    out[region] = int(cost.argmin())


def remove_small(labels, unary, min_area):
    """min_area: {class: pixels}. Returns new labels and how many regions were relabeled."""
    out = np.array(labels, dtype=np.int64)
    removed = 0
    for cls, area in min_area.items():
        comp, n = ndimage.label(out == cls, structure=EIGHT)
        if not n:
            continue
        sizes = np.bincount(comp.ravel())
        for r in np.flatnonzero(sizes[1:] < area) + 1:
            _relabel(out, comp == r, unary, cls)
            removed += 1
    return out, removed


def remove_unsupported(labels, unary, rules):
    """rules: [{"cls", "needs", "min_share"}]. Removes regions whose border lacks `needs`."""
    out = np.array(labels, dtype=np.int64)
    removed = 0
    for rule in rules:
        cls, needs, min_share = int(rule["cls"]), int(rule["needs"]), float(rule["min_share"])
        comp, n = ndimage.label(out == cls, structure=EIGHT)
        for r in range(1, n + 1):
            region = comp == r
            ring = ndimage.binary_dilation(region, structure=EIGHT) & ~region
            if ring.sum() and (out[ring] == needs).mean() < min_share:
                _relabel(out, region, unary, cls)
                removed += 1
    return out, removed


def remove_rare(labels, unary, min_share):
    """min_share: {class: fraction of the tile}. Removes a class that covers less."""
    out = np.array(labels, dtype=np.int64)
    removed = 0
    for cls, share in min_share.items():
        region = out == cls
        if region.any() and region.mean() < share:
            comp, n = ndimage.label(region, structure=EIGHT)
            for r in range(1, n + 1):
                _relabel(out, comp == r, unary, cls)
            removed += 1
    return out, removed


class MinRegionSolver:
    """
    inner:
        Solver run first.
    min_area, border, min_share:
        The region rules above; each is off when empty. Applied in this order.
    """

    def __init__(self, inner, min_area=None, border=None, min_share=None):
        self.inner = inner
        self.min_area = {int(k): int(v) for k, v in dict(min_area or {}).items() if v}
        if isinstance(border, dict):          # {name: rule}, so configs can --set one rule
            border = list(border.values())
        self.border = [dict(r) for r in (border or []) if float(dict(r).get("min_share", 0))]
        self.min_share = {int(k): float(v) for k, v in dict(min_share or {}).items() if v}

    def prepare(self, sample, prediction):
        if hasattr(self.inner, "prepare"):
            self.inner.prepare(sample, prediction)

    def solve(self, problem, init):
        start = time.perf_counter()
        result = self.inner.solve(problem, init)
        labels, small = remove_small(result.labels, problem.unary, self.min_area)
        labels, unsupported = remove_unsupported(labels, problem.unary, self.border)
        labels, rare = remove_rare(labels, problem.unary, self.min_share)
        out = SolveResult(labels, energy(problem, labels), time.perf_counter() - start,
                          trace=list(result.trace) + [{"stage": "min_region", "energy": energy(problem, labels)}])
        out.info = {**getattr(result, "info", {}), "small_regions_removed": small,
                    "unsupported_regions_removed": unsupported, "rare_classes_removed": rare}
        return out
