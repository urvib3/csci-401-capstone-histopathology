import json

import numpy as np
import pytest

from wcsp.config import ProblemConfig
from wcsp.data.base import Sample
from wcsp.labels import NUM_LABELS
from wcsp.model import build_problem
from wcsp.solve import ICMSolver, energy
from wcsp.solve.clusters import ClusterSolver, cluster_graph
from wcsp.solve.factory import get_solver
from wcsp.solve.presence import PresenceSolver
from wcsp.solve.regions import MinRegionSolver, remove_rare, remove_small, remove_unsupported


def random_problem(shape=(24, 24), seed=0, lam=8.0):
    rng = np.random.default_rng(seed)
    logits = 2 * rng.normal(size=(NUM_LABELS,) + shape)
    probs = np.exp(logits) / np.exp(logits).sum(axis=0)
    return build_problem(probs, ProblemConfig(lam=lam)), rng


def test_cluster_energy_equals_pixel_energy():
    problem, rng = random_problem()
    segments = rng.integers(0, 6, (4, 4)).repeat(6, 0).repeat(6, 1)   # 16 blocks, 6 ids
    segments = np.unique(segments, return_inverse=True)[1].reshape(segments.shape)
    unary, table_w, change_w = cluster_graph(problem, segments)
    z = rng.integers(0, NUM_LABELS, segments.max() + 1)
    pairs = sum(table_w[a, b] * problem.table[z[a], z[b]] for a in range(len(z)) for b in range(a + 1, len(z)))
    cluster_total = unary[np.arange(len(z)), z].sum() + pairs + problem.constant
    assert cluster_total == pytest.approx(energy(problem, z[segments]))


def test_cluster_solver_runs_and_refines():
    problem, rng = random_problem(shape=(32, 32))
    image = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    solver = get_solver({"name": "clusters", "n_segments": 20, "features": "both", "refine": {"name": "icm"}})
    solver.prepare(Sample(id="t", image=image, target=None), None)
    result = solver.solve(problem, problem.unary.argmin(-1))
    assert result.labels.shape == (32, 32)
    assert result.trace[-1]["energy"] <= result.trace[0]["energy"] + 1e-9   # refine never raises energy
    alone = ClusterSolver(n_segments=20, features="probs", compactness=1e4).solve(problem, problem.unary.argmin(-1))   # noise probs: keep it grid-like
    assert alone.info["clusters"] >= 2


def test_remove_small_regions():
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 3            # 4 px necrosis: removed
    labels[5:10, 5:10] = 3          # 25 px necrosis: kept
    unary = np.zeros((10, 10, NUM_LABELS))
    unary[..., 2] = -1              # stroma is the best alternative
    out, removed = remove_small(labels, unary, {3: 10})
    assert removed == 1 and np.all(out[1:3, 1:3] == 2) and np.all(out[5:, 5:] == 3)


def test_neighbor_rule_drops_unsupported_class(tmp_path):
    problem, _ = random_problem(shape=(8, 8), lam=0.0)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"cost": {str(m): 0.0 for m in range(1, 32)}}))
    nb_prior = tmp_path / "nb.json"
    nb_prior.write_text(json.dumps({"has": [0.9] * 5, "none": [0.9, 0.9, 0.9, 0.001, 0.9]}))
    neighbors = tmp_path / "neighbors.json"
    neighbors.write_text(json.dumps({"t": 0b10111}))      # neighbors have every class but necrosis
    solver = PresenceSolver(ICMSolver(), str(prior), gamma=0, neighbors=str(neighbors),
                            neighbor_prior=str(nb_prior), nb_gamma=1e4)
    solver.prepare(Sample(id="t", image=np.zeros((8, 8, 3), np.uint8), target=None), None)
    result = solver.solve(problem, problem.unary.argmin(-1))
    assert not result.info["present"] >> 3 & 1 and not np.any(result.labels == 3)


def test_remove_unsupported_border():
    labels = np.full((12, 12), 2, dtype=np.int64)    # stroma
    labels[:, 6:] = 0                                # tumor on the right
    labels[2:4, 2:4] = 1                             # inflammatory in stroma: kept
    labels[8:10, 8:10] = 1                           # inflammatory in tumor: removed
    unary = np.zeros((12, 12, NUM_LABELS))
    unary[..., 0] = -1
    out, removed = remove_unsupported(labels, unary, [{"cls": 1, "needs": 2, "min_share": 0.5}])
    assert removed == 1 and np.all(out[2:4, 2:4] == 1) and np.all(out[8:10, 8:10] == 0)


def test_remove_rare_class():
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[0, 0] = 4                                 # 1% Other: removed at 2%
    unary = np.zeros((10, 10, NUM_LABELS))
    out, removed = remove_rare(labels, unary, {4: 0.02})
    assert removed == 1 and not np.any(out == 4)


def test_region_rules_off_by_default():
    problem, _ = random_problem(shape=(16, 16))
    base = ICMSolver().solve(problem, problem.unary.argmin(-1))
    wrapped = MinRegionSolver(ICMSolver(), min_area={3: 0}, border={"infl": {"cls": 1, "needs": 2, "min_share": 0}},
                              min_share={4: 0}).solve(problem, problem.unary.argmin(-1))
    assert np.array_equal(base.labels, wrapped.labels)


def test_label_cost_drops_class(tmp_path):
    problem, _ = random_problem(shape=(8, 8), lam=0.0)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"cost": {str(m): 0.0 for m in range(1, 32)}}))
    solver = PresenceSolver(ICMSolver(), str(prior), gamma=0, label_cost={4: 1e4})
    result = solver.solve(problem, problem.unary.argmin(-1))
    assert not result.info["present"] >> 4 & 1 and not np.any(result.labels == 4)
