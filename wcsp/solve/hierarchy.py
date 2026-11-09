# Multi-scale block WCSP: the pixel grid plus coarser grids of blocks.
#
# Level 0 is the ordinary pixel problem. Level l >= 1 is a grid of
# s_l x s_l pixel blocks, s_l = size * 2^(l-1) (4, 8, 16 px by default), and
# every block is a variable with its own label. A block's children are the
# pixels inside it (level 1) or the four level l-1 blocks inside it.
#
#   E(x, z) = E_pixel(x)                                         as before
#           + sum_l sum_B  block_alpha * a_l * U_B(z_B)          block unary
#           + sum_l sum_{B,B'} s_l * w_BB' * block_lam * b[z_B, z_B']   block neighbors
#           + sum_l sum_B sum_{c child of B}  mu * a_c * [z_c != z_B]   parent-child
#
# U_B is -log of the classifier probabilities averaged over the block, a_l is
# a block's area in pixels and s_l its side, so every term is in "per pixel"
# units: a block's unary counts as much as its pixels together, a block
# boundary as much as the pixel pairs along it, and a parent-child mismatch
# as much as the pixels of the child.
#
# Solved by block coordinate descent over levels: fix every other level,
# fold the parent-child links into the unaries of one level, and solve that
# level as an ordinary GridProblem (the pixel level with the usual solver,
# the block levels with ICM). Each step is exact conditioning followed by a
# solver that never raises its own energy, so the total never increases.

import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..model.neighborhood import distance_weights, forward_offsets
from ..model.problem import GridProblem
from .base import SolveResult
from .energy import energy
from .icm import ICMSolver


@dataclass
class HierarchyResult(SolveResult):
    levels: list = field(default_factory=list)   # (h_l, w_l) block labels, level 1 first


def pixel_probs(problem):
    """Classifier probabilities recovered from -log p unaries (alpha = 1, T = 1)."""
    logits = -problem.unary
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    return probs / probs.sum(-1, keepdims=True)


def pool(array, factor):
    """Sum (H, W, ...) over non-overlapping factor x factor windows."""
    h, w = array.shape[:2]
    return array.reshape(h // factor, factor, w // factor, factor, *array.shape[2:]).sum((1, 3))


def upsample(array, factor):
    return np.repeat(np.repeat(array, factor, 0), factor, 1)


def one_hot(labels, k):
    return (labels[..., None] == np.arange(k)).astype(np.float64)


class HierarchySolver:
    """
    pixel:
        Solver for the pixel level (e.g. the 100/56/100 ICM block scheme).
    levels, size:
        Number of block levels; level 1 blocks are size x size pixels and
        each level doubles the side.
    mu:
        Parent-child mismatch cost per pixel of the child.
    block_alpha, block_lam, block_types, eta:
        Block unary weight, block neighbor weight (same b table as the
        pixels) and the block neighborhood (types as in wcsp.model.neighborhood).
    """

    def __init__(self, pixel, levels=3, size=4, mu=1.0, block_alpha=1.0, block_lam=8.0,
                 block_types=(1, 2), eta=1.0, max_rounds=4, tol=1e-9):
        if levels < 1:
            raise ValueError("levels must be >= 1.")
        self.pixel = pixel
        self.levels = levels
        self.size = size
        self.mu = mu
        self.block_alpha = block_alpha
        self.block_lam = block_lam
        self.block_types = tuple(block_types)
        self.eta = eta
        self.max_rounds = max_rounds
        self.tol = tol
        self.block_solver = ICMSolver()

    # --- building the block levels -------------------------------------------------

    def sides(self):
        return [self.size * 2 ** l for l in range(self.levels)]

    def block_problems(self, problem):
        """One GridProblem per block level, without the parent-child links."""
        if problem.active is not None:
            raise ValueError("the hierarchy needs a fully active pixel problem.")
        height, width = problem.shape
        top = self.sides()[-1]
        if height % top or width % top:
            raise ValueError(f"a {height} x {width} grid does not split into {top} px blocks.")
        probs = pixel_probs(problem)
        offsets = forward_offsets(self.block_types)
        base = distance_weights(offsets, self.eta, normalize=True)
        out = []
        for side in self.sides():
            mean = pool(probs, side) / side ** 2
            unary = -np.log(np.clip(mean, 1e-6, 1.0))
            unary -= unary.min(-1, keepdims=True)
            out.append(GridProblem(
                unary=self.block_alpha * side ** 2 * unary,
                offsets=offsets,
                weights=self.block_lam * side * base,
                table=problem.table,
            ))
        return out

    # --- energy --------------------------------------------------------------------

    def _child_area(self, level):
        """Pixels per child of a level (1-based) block."""
        return 1 if level == 1 else self.sides()[level - 2] ** 2

    def _factor(self, level):
        """Children per side of a level (1-based) block."""
        return self.size if level == 1 else 2

    def total_energy(self, problem, blocks, labels, block_labels):
        total = energy(problem, labels)
        below = labels
        for l, (sub, z) in enumerate(zip(blocks, block_labels), start=1):
            total += energy(sub, z)
            mismatch = below != upsample(z, self._factor(l))
            total += self.mu * self._child_area(l) * mismatch.sum()
            below = z
        return float(total)

    # --- one level given the others ------------------------------------------------

    def _link_costs(self, level, k, labels, block_labels):
        """(h, w, k) parent-child cost of each label at each variable of a level (0 = pixels)."""
        mu = self.mu
        if level == 0:
            parent = upsample(block_labels[0], self.size)
            return mu * (np.arange(k) != parent[..., None])
        below = labels if level == 1 else block_labels[level - 2]
        factor = self._factor(level)
        children = pool(one_hot(below, k), factor)                     # (h, w, k) counts
        cost = mu * self._child_area(level) * (factor ** 2 - children)
        if level < self.levels:
            parent = upsample(block_labels[level], 2)
            cost = cost + mu * self.sides()[level - 1] ** 2 * (np.arange(k) != parent[..., None])
        return cost

    def solve(self, problem, init):
        start = time.perf_counter()
        k = problem.num_labels
        blocks = self.block_problems(problem)
        labels = np.array(init, dtype=np.int64)

        # Start each block at its best label given the level below it.
        block_labels = []
        for l, sub in enumerate(blocks, start=1):
            below = labels if l == 1 else block_labels[-1]
            factor = self._factor(l)
            children = pool(one_hot(below, k), factor)
            cost = sub.unary + self.mu * self._child_area(l) * (factor ** 2 - children)
            block_labels.append(cost.argmin(-1))

        current = self.total_energy(problem, blocks, labels, block_labels)
        trace = [{"round": 0, "level": None, "energy": current, "seconds": 0.0}]
        for round_ in range(1, self.max_rounds + 1):
            round_start = current
            for level in range(self.levels + 1):
                if level == 0:
                    sub = replace(problem, unary=problem.unary + self._link_costs(0, k, labels, block_labels))
                    labels = self.pixel.solve(sub, labels).labels
                else:
                    base = blocks[level - 1]
                    sub = replace(base, unary=base.unary + self._link_costs(level, k, labels, block_labels))
                    block_labels[level - 1] = self.block_solver.solve(sub, block_labels[level - 1]).labels
                current = self.total_energy(problem, blocks, labels, block_labels)
                trace.append({"round": round_, "level": level, "energy": current,
                              "seconds": time.perf_counter() - start})
            if current >= round_start - self.tol:
                break

        return HierarchyResult(labels, current, time.perf_counter() - start, trace=trace,
                               levels=block_labels)
