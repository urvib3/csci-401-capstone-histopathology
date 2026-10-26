# Run several solvers one after another, each starting from the previous
# result, e.g. fast ICM over the whole grid, then toulbar2 on small tiles.

import time

from .base import SolveResult


class ChainSolver:
    def __init__(self, stages):
        self.stages = stages

    def solve(self, problem, init):
        start = time.perf_counter()
        labels, trace, result = init, [], None
        for i, stage in enumerate(self.stages):
            result = stage.solve(problem, labels)
            labels = result.labels
            trace.extend({**t, "stage": i} for t in result.trace)
        return SolveResult(labels, result.energy, time.perf_counter() - start,
                           lower_bound=result.lower_bound, trace=trace)
