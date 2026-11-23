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
#
# Neighbor-tile rule (optional): tissue compartments are larger than a tile, so a class
# is likely in a tile when an adjacent tile has it and very unlikely when none does
# (fit ground truth: necrosis 50% vs 0.7%). With a neighbors file (tile id -> classes the
# U-Net predicts in the 4 adjacent tiles) and P(k in tile | a neighbor has k / none does)
# from fit, each set pays nb_gamma * sum_k -log P(k in S or not | neighbors). The same
# mechanism takes ROI context instead (rules/rule_stats.py: classes in the ROI's other tiles).
#
# Label costs (optional): label_cost[k] is added to every set containing k, so classes the
# U-Net over-predicts (e.g. Other: 20% of tiles vs 12% in truth) can be made costlier alone.

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
    neighbors, neighbor_prior, nb_gamma:
        Optional neighbor-tile rule (see the header); off when nb_gamma = 0.
    """

    def __init__(self, inner, prior, gamma=100.0, forbid=1e3, neighbors=None, neighbor_prior=None,
                 nb_gamma=0.0, label_cost=None):
        self.inner = inner
        # label_cost: {class: cost} paid by every set containing the class (per-class
        # strength on top of gamma, as label costs in Delong et al. 2012).
        self.label_cost = {int(c): float(v) for c, v in dict(label_cost or {}).items() if v}   # 0 = off
        self.prior_path = prior
        self.gamma = gamma
        self.forbid = forbid
        self.nb_gamma = nb_gamma
        self.neighbors = json.loads(Path(neighbors).read_text()) if neighbors else {}
        self.nb_prior = json.loads(Path(neighbor_prior).read_text()) if neighbor_prior else None
        if nb_gamma and (not self.neighbors or self.nb_prior is None):
            raise ValueError("nb_gamma needs both a neighbors file and a neighbor prior.")
        self.sample_id = None

    def prepare(self, sample, prediction):
        self.sample_id = sample.id
        if hasattr(self.inner, "prepare"):
            self.inner.prepare(sample, prediction)

    def neighbor_costs(self, k):
        """(2^K,) neighbor-rule cost of every class set for the current tile (zeros if unknown)."""
        out = np.zeros(2 ** k)
        mask = self.neighbors.get(self.sample_id) if self.nb_gamma else None
        if mask is None or mask < 0:
            return out
        p = np.array([self.nb_prior["has" if mask >> c & 1 else "none"][c] for c in range(k)])
        p = np.clip(p, 0.005, 0.995)
        for s in range(1, 2 ** k):
            out[s] = sum(-np.log(p[c]) if s >> c & 1 else -np.log(1 - p[c]) for c in range(k))
        return out

    def solve(self, problem, init):
        start = time.perf_counter()
        k = problem.num_labels
        prior = load_prior(self.prior_path, k)
        active = problem.active_mask
        costs = set_costs(problem.unary[active][None], prior, self.gamma)
        costs = costs + self.nb_gamma * self.neighbor_costs(k)
        for c, v in self.label_cost.items():
            costs = costs + v * np.array([s >> c & 1 for s in range(2 ** k)])
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
