# The alternating block scheme from the project notes.
#
# A 256 x 256 image is split into four 100 x 100 corner blocks and the
# 56-pixel-wide cross between them:
#
#     +-----+--+-----+
#     |  A  |  |  B  |       phase 1: fix the cross, solve A, B, C, D
#     +-----+  +-----+                (independent: the cross is wider than
#     |    cross     |                 any interaction, so they can run in
#     +-----+  +-----+                 parallel)
#     |  C  |  |  D  |       phase 2: fix the corners, solve the cross
#     +-----+--+-----+
#
# Rounds repeat until the energy stops falling or time runs out. Each region
# is solved by an inner solver on a conditioned subproblem, and its result is
# kept only if it lowers the energy, so the total never increases whatever
# the inner solver does (e.g. a toulbar2 run cut off by its time limit).

import time
from dataclasses import dataclass

import numpy as np

from .base import SolveResult
from .condition import condition
from .energy import energy

# Largest offset coordinate of any neighbor (type 5 is (2, 2)).
REACH = 2


@dataclass(frozen=True)
class BlockLayout:
    """phases[i] is a tuple of (H, W) bool masks solved independently in phase i."""

    phases: tuple

    @classmethod
    def cross(cls, shape, block=100):
        """Four corner blocks of block x block pixels and the cross between them."""
        height, width = shape
        band = min(height, width) - 2 * block
        if band <= REACH:
            raise ValueError(
                f"the cross is {band} px wide; it must be wider than {REACH} px "
                "so the corner blocks do not interact."
            )
        corners = []
        for rows in (slice(0, block), slice(height - block, height)):
            for cols in (slice(0, block), slice(width - block, width)):
                mask = np.zeros(shape, dtype=bool)
                mask[rows, cols] = True
                corners.append(mask)
        cross = ~np.any(corners, axis=0)
        return cls(phases=(tuple(corners), (cross,)))

    @classmethod
    def tiles(cls, shape, size=16):
        """
        size x size tiles REACH px apart, so the tiles of one phase never
        interact; four phases shifted by half a stride cover every pixel.
        """
        height, width = shape
        stride = size + REACH
        phases = []
        for dy, dx in ((0, 0), (stride // 2, 0), (0, stride // 2), (stride // 2, stride // 2)):
            regions = []
            for y in range(dy - stride, height, stride):
                for x in range(dx - stride, width, stride):
                    rows = slice(max(y, 0), max(min(y + size, height), 0))
                    cols = slice(max(x, 0), max(min(x + size, width), 0))
                    if rows.stop > rows.start and cols.stop > cols.start:
                        mask = np.zeros(shape, dtype=bool)
                        mask[rows, cols] = True
                        regions.append(mask)
            phases.append(tuple(regions))
        return cls(phases=tuple(phases))

    @classmethod
    def whole(cls, shape):
        """No decomposition: one region covering the image."""
        return cls(phases=((np.ones(shape, dtype=bool),),))


def _solve_region(inner, sub, init):
    return inner.solve(sub, init)


class BlockCoordinateDescent:
    """
    inner:
        Solver used on each region (ICMSolver, Toulbar2Solver, ...).
    layout:
        BlockLayout, a function shape -> BlockLayout, or None for
        BlockLayout.cross(problem.shape).
    time_limit:
        Seconds; checked between phases.
    workers:
        Processes used to solve the regions of one phase in parallel.
    """

    def __init__(self, inner, layout=None, max_rounds=10, time_limit=None, tol=1e-9,
                 workers=1):
        self.inner = inner
        self.layout = layout
        self.max_rounds = max_rounds
        self.time_limit = time_limit
        self.tol = tol
        self.workers = workers

    def _layout(self, shape):
        if callable(self.layout):
            return self.layout(shape)
        return self.layout or BlockLayout.cross(shape)

    def solve(self, problem, init):
        start = time.perf_counter()
        layout = self._layout(problem.shape)
        labels = np.array(init, dtype=np.int64)
        active = problem.active_mask
        current = energy(problem, labels)
        trace = [{"round": 0, "phase": None, "energy": current, "seconds": 0.0}]
        pool = None
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            pool = ProcessPoolExecutor(self.workers)

        def out_of_time():
            return self.time_limit and time.perf_counter() - start > self.time_limit

        try:
            for round_ in range(1, self.max_rounds + 1):
                round_start = current
                for phase, regions in enumerate(layout.phases):
                    if out_of_time():
                        break
                    # Regions of one phase never interact, so they are all
                    # conditioned on the same labels and solved independently.
                    jobs = []
                    for region in regions:
                        region = region & active
                        if region.any():
                            sub, window = condition(problem, labels, region)
                            before = labels[window].copy()
                            # Pairwise costs are >= 0, so the sum of the smallest
                            # unaries bounds the region's optimum: skip regions
                            # already at that bound.
                            bound = sub.constant + sub.unary[sub.active_mask].min(-1).sum()
                            if energy(sub, before) > bound + self.tol:
                                jobs.append((sub, window, before))
                    if pool is None:
                        results = [self.inner.solve(sub, before) for sub, _, before in jobs]
                    else:
                        futures = [pool.submit(_solve_region, self.inner, sub, before)
                                   for sub, _, before in jobs]
                        results = [f.result() for f in futures]
                    for (sub, window, before_labels), result in zip(jobs, results):
                        if result.energy < energy(sub, before_labels) - self.tol:
                            labels[window] = np.where(sub.active, result.labels, labels[window])
                    current = energy(problem, labels)
                    trace.append({
                        "round": round_, "phase": phase, "energy": current,
                        "seconds": time.perf_counter() - start,
                    })
                if current >= round_start - self.tol or out_of_time():
                    break
        finally:
            if pool is not None:
                pool.shutdown()

        return SolveResult(labels, current, time.perf_counter() - start, trace=trace)
