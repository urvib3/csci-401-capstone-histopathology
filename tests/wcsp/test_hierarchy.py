import itertools

import numpy as np
import pytest

from wcsp.config import ProblemConfig
from wcsp.labels import NUM_LABELS
from wcsp.model import build_problem
from wcsp.solve import ICMSolver, energy
from wcsp.solve.factory import get_solver
from wcsp.solve.hierarchy import HierarchySolver


def random_problem(shape=(16, 16), seed=0, lam=8.0):
    rng = np.random.default_rng(seed)
    logits = 2 * rng.normal(size=(NUM_LABELS,) + shape)
    probs = np.exp(logits) / np.exp(logits).sum(axis=0)
    return build_problem(probs, ProblemConfig(lam=lam)), probs, rng


def brute_total(solver, problem, probs, labels, block_labels):
    """The hierarchy energy term by term, straight from the formula in hierarchy.py."""
    total = energy(problem, labels)
    table = problem.table
    below, below_side = labels, 1
    for side, z in zip(solver.sides(), block_labels):
        h, w = z.shape
        for by, bx in itertools.product(range(h), range(w)):
            window = probs[:, by * side:(by + 1) * side, bx * side:(bx + 1) * side]
            unary = -np.log(np.clip(window.mean((1, 2)), 1e-6, 1))
            total += solver.block_alpha * side ** 2 * (unary[z[by, bx]] - unary.min())
            # children: every below-level variable inside the block
            f = side // below_side
            for cy, cx in itertools.product(range(by * f, (by + 1) * f), range(bx * f, (bx + 1) * f)):
                total += solver.mu * below_side ** 2 * (below[cy, cx] != z[by, bx])
        # 8-neighborhood with 1/d weights, normalized as for pixels
        norm = 2 * (2 * 1 + 2 / np.sqrt(2))
        for a, b in itertools.combinations(itertools.product(range(h), range(w)), 2):
            d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d2 <= 2:
                total += solver.block_lam * side * table[z[a], z[b]] / np.sqrt(d2) / norm
        below, below_side = z, side
    return total


def test_energy_matches_brute_force():
    problem, probs, rng = random_problem()
    solver = HierarchySolver(ICMSolver(), levels=2, size=4, mu=0.7, block_alpha=0.5, block_lam=3.0)
    labels = rng.integers(0, NUM_LABELS, problem.shape)
    block_labels = [rng.integers(0, NUM_LABELS, (4, 4)), rng.integers(0, NUM_LABELS, (2, 2))]
    blocks = solver.block_problems(problem)
    got = solver.total_energy(problem, blocks, labels, block_labels)
    assert got == pytest.approx(brute_total(solver, problem, probs, labels, block_labels))


def test_never_increases_energy():
    problem, _, rng = random_problem(shape=(32, 32), lam=20.0)
    solver = HierarchySolver(ICMSolver(), levels=3, size=4, mu=0.5)
    init = rng.integers(0, NUM_LABELS, problem.shape)
    result = solver.solve(problem, init)
    energies = [t["energy"] for t in result.trace]
    assert np.all(np.diff(energies) <= 1e-9)
    blocks = solver.block_problems(problem)
    assert result.energy == pytest.approx(solver.total_energy(problem, blocks, result.labels, result.levels))
    assert [z.shape for z in result.levels] == [(8, 8), (4, 4), (2, 2)]


def test_mu_zero_gives_the_pixel_solution():
    # With no parent-child links the blocks cannot affect the pixels.
    problem, _, _ = random_problem(shape=(32, 32), lam=20.0)
    init = problem.unary.argmin(-1)
    alone = ICMSolver().solve(problem, init).labels
    result = HierarchySolver(ICMSolver(), levels=3, mu=0.0).solve(problem, init)
    assert np.array_equal(result.labels, alone)


def test_strong_links_make_blocks_uniform():
    problem, _, _ = random_problem(shape=(16, 16), lam=1.0)
    result = HierarchySolver(ICMSolver(), levels=1, size=4, mu=100.0).solve(problem, problem.unary.argmin(-1))
    blocks = result.labels.reshape(4, 4, 4, 4).transpose(0, 2, 1, 3).reshape(16, 16)
    assert np.all(blocks == blocks[:, :1])


def test_factory_and_size_check():
    solver = get_solver({"name": "hierarchy", "levels": 2, "mu": 1.0,
                         "pixel": {"name": "icm"}})
    assert isinstance(solver, HierarchySolver) and solver.sides() == [4, 8]
    problem, _, _ = random_problem(shape=(12, 12))
    with pytest.raises(ValueError):
        solver.block_problems(problem)
