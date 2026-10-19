# Classifier-only baseline: each pixel takes its cheapest unary label.

import time

import numpy as np

from .base import SolveResult
from .energy import energy


class ArgmaxSolver:
    """Ignores the pairwise rules; equals the classifier's own prediction."""

    def solve(self, problem, init):
        start = time.perf_counter()
        labels = np.array(init, dtype=np.int64)
        active = problem.active_mask
        labels[active] = problem.unary.argmin(axis=-1)[active]
        return SolveResult(labels, energy(problem, labels), time.perf_counter() - start)
