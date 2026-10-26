import itertools

import numpy as np
import pytest

from wcsp.config import ProblemConfig
from wcsp.labels import NUM_LABELS
from wcsp.model import build_problem
from wcsp.solve import (
    ArgmaxSolver, BlockCoordinateDescent, BlockLayout, ICMSolver, condition, energy,
    local_costs,
)


def random_problem(shape=(6, 7), seed=0, **settings):
    rng = np.random.default_rng(seed)
    logits = 2 * rng.normal(size=(NUM_LABELS,) + shape)
    probs = np.exp(logits) / np.exp(logits).sum(axis=0)
    image = rng.integers(0, 255, shape + (3,), dtype=np.uint8)
    settings = {"contrast_beta": 0.7, "contrast_sigma": 0.5, **settings}
    return build_problem(probs, ProblemConfig(**settings), image=image), image, rng


def brute_energy(problem, labels, image, config):
    """Pair-by-pair cost straight from the formula in the notes."""
    from wcsp.model import load_table

    table = load_table(config.table)
    height, width = labels.shape
    z = image.astype(float) / 255
    total = sum(problem.unary[y, x, labels[y, x]] for y in range(height) for x in range(width))
    pixels = list(itertools.product(range(height), range(width)))
    norm = 2 * sum(np.hypot(*o) ** -config.eta for o in problem.offsets)
    for a, b in itertools.combinations(pixels, 2):
        d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        if d2 > 8:
            continue
        i, j = labels[a], labels[b]
        f = config.contrast_beta * np.exp(-((z[a] - z[b]) ** 2).sum() / (2 * config.contrast_sigma ** 2))
        cost = table[i, j] + f * (i != j)
        total += config.lam * cost / d2 ** (config.eta / 2) / norm
    return total


def test_energy_matches_brute_force():
    config = ProblemConfig(contrast_beta=0.7, contrast_sigma=0.5, lam=3.0, eta=1.5)
    problem, image, rng = random_problem(lam=3.0, eta=1.5)
    labels = rng.integers(0, NUM_LABELS, problem.shape)
    assert energy(problem, labels) == pytest.approx(brute_energy(problem, labels, image, config))


def test_local_costs_equal_energy_differences():
    problem, _, rng = random_problem()
    labels = rng.integers(0, NUM_LABELS, problem.shape)
    costs = local_costs(problem, labels)
    base = energy(problem, labels)
    for _ in range(10):
        y, x, k = rng.integers(problem.shape[0]), rng.integers(problem.shape[1]), rng.integers(NUM_LABELS)
        changed = labels.copy()
        changed[y, x] = k
        expected = costs[y, x, k] - costs[y, x, labels[y, x]]
        assert energy(problem, changed) - base == pytest.approx(expected)


def test_condition_preserves_energy():
    problem, _, rng = random_problem(shape=(9, 10))
    labels = rng.integers(0, NUM_LABELS, problem.shape)
    region = rng.random(problem.shape) < 0.4         # irregular region
    sub, window = condition(problem, labels, region)
    assert sub.num_variables == region.sum()
    for _ in range(5):
        trial = labels.copy()
        trial[region] = rng.integers(0, NUM_LABELS, region.sum())
        assert energy(sub, trial[window]) == pytest.approx(energy(problem, trial))

    # Conditioning a conditioned problem again stays consistent.
    inner = sub.active & (rng.random(sub.shape) < 0.5)
    inner[np.argwhere(sub.active)[0][0], np.argwhere(sub.active)[0][1]] = True
    sub2, window2 = condition(sub, labels[window], inner)
    assert energy(sub2, labels[window][window2]) == pytest.approx(energy(problem, labels))
    with pytest.raises(ValueError):
        condition(sub, labels[window], ~sub.active)


def test_huge_alpha_reproduces_the_classifier():
    # Workflow step 1 in the notes: with fidelity cranked up, every solver
    # must return exactly the classifier's argmax.
    problem, _, _ = random_problem(shape=(24, 24), alpha=1e4)
    target = problem.unary.argmin(-1)
    init = np.zeros(problem.shape, dtype=np.int64)
    layout = BlockLayout.cross(problem.shape, block=9)
    for solver in (ArgmaxSolver(), ICMSolver(), BlockCoordinateDescent(ICMSolver(), layout)):
        assert np.array_equal(solver.solve(problem, init).labels, target)


def test_icm_reaches_a_local_minimum():
    problem, _, rng = random_problem(shape=(16, 16), lam=20.0)
    init = problem.unary.argmin(-1)
    result = ICMSolver().solve(problem, init)
    assert np.all(np.diff(result.trace) <= 1e-9)
    assert result.energy <= energy(problem, init)
    costs = local_costs(problem, result.labels)
    current = np.take_along_axis(costs, result.labels[..., None], -1)[..., 0]
    assert np.all(costs.min(-1) >= current - 1e-9)


def test_cross_layout_matches_the_notes():
    layout = BlockLayout.cross((256, 256), block=100)
    corners, (cross,) = layout.phases
    assert [c.sum() for c in corners] == [10000] * 4
    assert cross.sum() == 256 * 56 * 2 - 56 * 56
    assert np.array_equal(np.sum(corners, axis=0) + cross, np.ones((256, 256)))
    with pytest.raises(ValueError):
        BlockLayout.cross((10, 10), block=4)


def test_tile_layout_covers_every_pixel_without_interacting_tiles():
    from wcsp.solve.blocks import REACH

    layout = BlockLayout.tiles((40, 37), size=8)
    covered = np.zeros((40, 37), dtype=bool)
    for regions in layout.phases:
        for a, b in itertools.combinations(regions, 2):
            ya, xa = np.nonzero(a)
            yb, xb = np.nonzero(b)
            gap_y = max(yb.min() - ya.max(), ya.min() - yb.max())
            gap_x = max(xb.min() - xa.max(), xa.min() - xb.max())
            assert max(gap_y, gap_x) > REACH
        covered |= np.any(regions, axis=0)
    assert covered.all()


def test_chain_never_increases_energy():
    from wcsp.solve.factory import get_solver

    problem, _, rng = random_problem(shape=(30, 30), lam=20.0)
    init = rng.integers(0, NUM_LABELS, problem.shape)
    solver = get_solver({"name": "chain", "stages": [
        {"name": "blocks", "block": 12, "inner": {"name": "icm"}},
        {"name": "blocks", "tile": 8, "inner": {"name": "icm"}},
    ]})
    first = solver.stages[0].solve(problem, init)
    result = solver.solve(problem, init)
    assert result.energy <= first.energy + 1e-9
    assert result.energy == pytest.approx(energy(problem, result.labels))


def test_block_descent_never_increases_energy():
    problem, _, rng = random_problem(shape=(30, 30), lam=20.0)
    init = rng.integers(0, NUM_LABELS, problem.shape)
    solver = BlockCoordinateDescent(ICMSolver(max_sweeps=2), BlockLayout.cross(problem.shape, block=12))
    result = solver.solve(problem, init)
    energies = [t["energy"] for t in result.trace]
    assert np.all(np.diff(energies) <= 1e-9)
    assert result.energy == pytest.approx(energy(problem, result.labels))
    assert result.energy < energies[0]


def test_block_descent_respects_fixed_pixels():
    problem, _, rng = random_problem(shape=(20, 20), lam=20.0)
    init = rng.integers(0, NUM_LABELS, problem.shape)
    region = np.zeros(problem.shape, dtype=bool)
    region[5:15, 3:18] = True
    region[8:11, 8:12] = False  # a hole of fixed pixels inside the window
    sub, window = condition(problem, init, region)
    layout = BlockLayout.whole(sub.shape)
    result = BlockCoordinateDescent(ICMSolver(), layout).solve(sub, init[window])
    assert np.array_equal(result.labels[~sub.active], init[window][~sub.active])
