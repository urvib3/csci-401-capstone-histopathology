import json

import numpy as np

from wcsp.config import ProblemConfig
from wcsp.labels import NUM_LABELS
from wcsp.model import build_problem
from wcsp.solve import ICMSolver
from wcsp.solve.factory import get_solver
from wcsp.solve.presence import PresenceSolver, best_set, set_costs


def problem_with(probs):
    return build_problem(probs, ProblemConfig(lam=4.0))


def write_prior(tmp_path, rare_mask, rare_cost=20.0, common_cost=1.0):
    cost = {str(m): (rare_cost if m & rare_mask == rare_mask else common_cost)
            for m in range(1, 2 ** NUM_LABELS)}
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"cost": cost}))
    return path


def two_class_tile(minor, major=2, size=16, patch=3, confidence=0.6):
    """Mostly `major`, with a small `minor` patch the U-Net is unsure about."""
    probs = np.full((NUM_LABELS, size, size), 0.05)
    probs[major] = 0.8
    probs[:, :patch, :patch] = 0.1
    probs[minor, :patch, :patch] = confidence
    return probs / probs.sum(0)


def test_set_costs_match_brute_force():
    rng = np.random.default_rng(0)
    unary = rng.random((1, 7, NUM_LABELS))
    prior = rng.random(2 ** NUM_LABELS)
    costs = set_costs(unary, prior, gamma=3.0)
    for mask in (1, 5, 31):
        allowed = [c for c in range(NUM_LABELS) if mask >> c & 1]
        assert np.isclose(costs[mask], unary[0][:, allowed].min(1).sum() + 3.0 * prior[mask])
    assert np.isinf(costs[0])


def test_ties_pick_the_largest_set():
    costs = np.full(2 ** NUM_LABELS, 5.0)
    costs[0] = np.inf
    assert best_set(costs) == 2 ** NUM_LABELS - 1


def test_gamma_zero_reproduces_the_inner_solver(tmp_path):
    problem = problem_with(two_class_tile(minor=3))
    init = problem.unary.argmin(-1)
    alone = ICMSolver().solve(problem, init).labels
    solver = PresenceSolver(ICMSolver(), write_prior(tmp_path, rare_mask=(1 << 3) | (1 << 2)), gamma=0.0)
    result = solver.solve(problem, init)
    assert np.array_equal(result.labels, alone)
    assert result.info["present"] == 2 ** NUM_LABELS - 1


def test_rare_combination_is_dropped_when_evidence_is_weak(tmp_path):
    # necrosis (3) inside stroma (2) is made rare; the weak necrosis patch goes away.
    problem = problem_with(two_class_tile(minor=3))
    prior = write_prior(tmp_path, rare_mask=(1 << 3) | (1 << 2))
    result = PresenceSolver(ICMSolver(), prior, gamma=1.0).solve(problem, problem.unary.argmin(-1))
    assert not (result.labels == 3).any()
    assert result.info["dropped"] == 1 << 3


def test_strong_evidence_beats_the_prior(tmp_path):
    probs = two_class_tile(minor=3, patch=8, confidence=0.95)
    problem = problem_with(probs)
    prior = write_prior(tmp_path, rare_mask=(1 << 3) | (1 << 2), rare_cost=5.0)
    result = PresenceSolver(ICMSolver(), prior, gamma=1.0).solve(problem, problem.unary.argmin(-1))
    assert (result.labels == 3).sum() > 32


def test_factory(tmp_path):
    solver = get_solver({"name": "presence", "gamma": 10, "prior": str(write_prior(tmp_path, 1)),
                         "inner": {"name": "icm"}})
    assert isinstance(solver, PresenceSolver) and isinstance(solver.inner, ICMSolver)
