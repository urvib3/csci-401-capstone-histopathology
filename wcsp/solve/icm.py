# Iterated conditional modes: greedy pixel-by-pixel descent.
#
# A fast local-search baseline and sanity check for toulbar2. Pixels whose
# coordinates agree modulo 3 are at least 3 apart, beyond the largest
# interaction distance (2 * sqrt(2)), so each of the 9 classes can be
# updated at once and every update still lowers (or keeps) the energy.

import time

import numpy as np

from .base import SolveResult
from .energy import energy, local_costs

STRIDE = 3


class ICMSolver:
    def __init__(self, max_sweeps=50, time_limit=None):
        self.max_sweeps = max_sweeps
        self.time_limit = time_limit

    def solve(self, problem, init):
        if np.abs(problem.offsets).max(initial=0) >= STRIDE:
            raise ValueError(f"ICM assumes neighbors closer than {STRIDE} pixels.")
        start = time.perf_counter()
        labels = np.array(init, dtype=np.int64)
        active = problem.active_mask
        rows, cols = np.indices(problem.shape)
        trace = [energy(problem, labels)]

        for _ in range(self.max_sweeps):
            changed = 0
            for dy in range(STRIDE):
                for dx in range(STRIDE):
                    batch = active & (rows % STRIDE == dy) & (cols % STRIDE == dx)
                    costs = local_costs(problem, labels)
                    best = costs.argmin(axis=-1)
                    current = np.take_along_axis(costs, labels[..., None], -1)[..., 0]
                    better = batch & (costs.min(axis=-1) < current - 1e-12)
                    labels[better] = best[better]
                    changed += int(better.sum())
            trace.append(energy(problem, labels))
            timed_out = self.time_limit and time.perf_counter() - start > self.time_limit
            if not changed or timed_out:
                break

        return SolveResult(labels, trace[-1], time.perf_counter() - start, trace=trace)
