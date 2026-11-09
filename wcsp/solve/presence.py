# Tile-level class presence: decide which classes a tile contains, then label pixels.
#
# A set S of classes is chosen per tile by
#
#   S* = argmin_S  sum_p min_{k in S} U_p(k)  +  gamma * cost(S)
#
# where U_p are the pixel unaries (-log p) and cost(S) = -log P(S) is how rare the
# combination is in the fit split's ground truth (fit_presence_prior.py). The first
# term is the U-Net's evidence: dropping a class costs what its pixels would pay for
# their best remaining class. The second encodes co-occurrence rules, e.g. necrosis
# and inflammatory almost never share a tile. Every set is enumerated (31 for 5
# classes), so the choice is exact.
#
# The inner solver (the usual WCSP) then runs with classes outside S* forbidden.
# With gamma = 0 the tie-break picks the largest set, so it reproduces the inner solver.

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def load_prior(path, num_labels):
    values = json.loads(Path(path).read_text())
    cost = np.full(2 ** num_labels, np.inf)
    for mask, value in values["cost"].items():
        cost[int(mask)] = value
    return cost


def set_costs(unary, prior, gamma):
    """(2^K,) cost of every class set; the empty set is infinite."""
    k = unary.shape[-1]
    flat = unary.reshape(-1, k)
    out = np.full(2 ** k, np.inf)
    for mask in range(1, 2 ** k):
        allowed = [c for c in range(k) if mask >> c & 1]
        out[mask] = flat[:, allowed].min(1).sum() + gamma * prior[mask]
    return out


def best_set(costs):
    """Lowest cost; ties go to the larger set."""
    order = sorted(range(1, len(costs)), key=lambda m: (costs[m], -bin(m).count("1")))
    return order[0]


class PresenceSolver:
    """
    inner:
        Solver run after the classes are chosen.
    prior:
        Path to the fit-split presence prior (JSON).
    gamma:
        Weight of the prior, in the units of the unaries (summed over pixels).
    forbid:
        Unary cost added to classes outside the chosen set.
    """

    def __init__(self, inner, prior, gamma=100.0, forbid=1e3):
        self.inner = inner
        self.prior_path = prior
        self.gamma = gamma
        self.forbid = forbid

    def solve(self, problem, init):
        start = time.perf_counter()
        k = problem.num_labels
        prior = load_prior(self.prior_path, k)
        active = problem.active_mask
        costs = set_costs(problem.unary[active][None], prior, self.gamma)
        chosen = best_set(costs)
        allowed = np.array([chosen >> c & 1 for c in range(k)], dtype=bool)
        unary = problem.unary + self.forbid * ~allowed
        sub = replace(problem, unary=unary)
        start_labels = np.array(init, dtype=np.int64)
        start_labels[active] = unary.argmin(-1)[active]
        result = self.inner.solve(sub, start_labels)
        argmax_mask = int(np.bitwise_or.reduce(1 << np.unique(problem.unary.argmin(-1)[active])))
        info = {"present": chosen, "argmax_present": argmax_mask,
                "dropped": int(argmax_mask & ~chosen)}
        result.info = {**getattr(result, "info", {}), **info}
        result.seconds = time.perf_counter() - start
        return result
