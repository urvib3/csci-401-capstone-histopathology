# Exact / anytime WCSP solving with toulbar2 (pytoulbar2).
#
# Each active pixel becomes a variable over the labels, with its unary cost
# table; each active neighboring pair gets a K x K binary cost function.
# toulbar2 searches with the current labeling's cost as its initial upper
# bound, so anything it returns is strictly better than where it started;
# when the time limit hits it returns the best labeling found so far (or
# nothing, and the starting labeling is kept).

import time

import numpy as np

from .base import SolveResult
from .energy import energy, partial_energy
from ..model.neighborhood import pair_slices


def pair_tables(problem):
    """
    Every active neighboring pair and its flattened K x K cost table.

    returns:
        (first, second, costs): flat pixel indices of each pair and an
        (n_pairs, K*K) float array; costs[n, i*K + j] is the cost of
        (label i at first[n], label j at second[n]).
    """
    height, width = problem.shape
    k = problem.num_labels
    ids = np.arange(height * width).reshape(height, width)
    active = problem.active_mask
    unlike = (1.0 - np.eye(k)).ravel()
    firsts, seconds, tables = [], [], []
    for o, offset in enumerate(problem.offsets):
        p, q = pair_slices(problem.shape, offset)
        both = active[p] & active[q]
        if not both.any():
            continue
        firsts.append(ids[p][both])
        seconds.append(ids[q][both])
        base = problem.weights[o] * problem.table.ravel()
        if problem.change is None:
            tables.append(np.broadcast_to(base, (int(both.sum()), k * k)))
        else:
            change = problem.change[o][p][both]
            tables.append(base + problem.weights[o] * change[:, None] * unlike)
    if not firsts:
        return np.empty(0, int), np.empty(0, int), np.empty((0, k * k))
    return np.concatenate(firsts), np.concatenate(seconds), np.concatenate(tables)


class Toulbar2Solver:
    """
    time_limit:
        Seconds per call (toulbar2 is anytime: it returns its best solution
        when the limit hits).
    resolution:
        Decimal digits kept when toulbar2 turns float costs into integers.
    options:
        Extra settings passed to pytoulbar2.CFN, e.g. {"vns": -1} for
        variable neighborhood search.
    """

    def __init__(self, time_limit=20, resolution=3, verbose=-1, options=None):
        self.time_limit = time_limit
        self.resolution = resolution
        self.verbose = verbose
        self.options = options or {}

    def build(self, problem, init):
        import pytoulbar2

        active = problem.active_mask
        flat = np.flatnonzero(active.ravel())
        var_of = -np.ones(active.size, dtype=np.int64)
        var_of[flat] = np.arange(len(flat))

        # Upper bound: the starting labeling's cost (without the constant),
        # plus a little slack for rounding to `resolution` digits.
        start_cost = partial_energy(problem, init, active)
        slack = len(flat) * 10.0 ** -self.resolution
        cfn = pytoulbar2.CFN(start_cost + slack, resolution=self.resolution,
                             verbose=self.verbose, **self.options)
        k = problem.num_labels
        cfn.AddVariables(len(flat), "x", 0, k - 1)
        unary = problem.unary.reshape(-1, k)[flat]
        cfn.AddFunctions(np.arange(len(flat)), np.ascontiguousarray(unary))
        first, second, costs = pair_tables(problem)
        if len(first):
            scopes = np.stack([var_of[first], var_of[second]], axis=1)
            cfn.AddFunctions(scopes, np.ascontiguousarray(costs.reshape(-1, k, k)))
        return cfn, flat

    def solve(self, problem, init):
        start = time.perf_counter()
        labels = np.array(init, dtype=np.int64)
        cfn, flat = self.build(problem, labels)
        build_seconds = time.perf_counter() - start
        result = cfn.Solve(showSolutions=0, timeLimit=int(np.ceil(self.time_limit)))
        lower_bound = None
        if result:
            solution = np.asarray(result[0], dtype=np.int64)
            candidate = labels.copy()
            candidate.ravel()[flat] = solution
            if energy(problem, candidate) < energy(problem, labels):
                labels = candidate
        try:
            lower_bound = problem.constant + float(cfn.GetDDualBound())
        except Exception:  # older pytoulbar2 versions
            pass
        return SolveResult(
            labels, energy(problem, labels), time.perf_counter() - start,
            lower_bound=lower_bound,
            trace=[{"build_seconds": build_seconds, "found": bool(result)}],
        )
