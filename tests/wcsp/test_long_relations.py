from dataclasses import replace
import itertools

import numpy as np
import pytest

from wcsp.config import ProblemConfig
from wcsp.model.neighborhood import sample_long_edges
from wcsp.model.problem import build_problem
from wcsp.solve.condition import condition
from wcsp.solve.energy import energy, local_costs
from wcsp.solve.factory import get_solver
from wcsp.solve.toulbar2 import pair_tables


def problem():
    rng = np.random.default_rng(12)
    probs = rng.dirichlet(np.ones(5), size=(10, 9)).transpose(2, 0, 1)
    return build_problem(probs, ProblemConfig(lam=10, long_probability=.15, long_distance=3)), rng


def test_sampler_exact_dense_and_reproducible():
    shape = (5, 6)
    edges = sample_long_edges(shape, 3, 1)
    expected = [(a, b) for a, b in itertools.combinations(range(30), 2)
                if (a//6-b//6)**2 + (a%6-b%6)**2 > 9]
    np.testing.assert_array_equal(edges, expected)
    for p in (0, .01, .25, 1):
        a = sample_long_edges(shape, 3, p, 42)
        np.testing.assert_array_equal(a, sample_long_edges(shape, 3, p, 42))
        assert len(np.unique(a, axis=0)) == len(a)
    with pytest.raises(ValueError):
        sample_long_edges((512, 512), 8, .01)
    with pytest.raises(ValueError):
        sample_long_edges(shape, 3, 1.1)


def test_sparse_energy_tables_and_local_deltas_agree():
    pr, rng = problem()
    labels = rng.integers(5, size=pr.shape)
    a, b, tables = pair_tables(pr)
    flat = labels.ravel()
    unary = np.take_along_axis(pr.unary, labels[..., None], -1).sum()
    assert energy(pr, labels) == pytest.approx(unary + tables[np.arange(len(a)), flat[a]*5+flat[b]].sum())
    costs = local_costs(pr, labels)
    before = energy(pr, labels)
    for y, x, k in ((0, 0, 2), (5, 5, 4), (9, 8, 0)):
        candidate = labels.copy()
        candidate[y, x] = k
        assert energy(pr, candidate)-before == pytest.approx(costs[y, x, k]-costs[y, x, labels[y, x]])


def test_conditioning_and_nested_conditioning_preserve_sparse_energy():
    pr, rng = problem()
    labels = rng.integers(5, size=pr.shape)
    region = np.zeros(pr.shape, bool)
    region[2:9, 1:8] = True
    region[4, 4] = False
    sub, window = condition(pr, labels, region)
    for _ in range(3):
        candidate = labels.copy()
        candidate[region] = rng.integers(5, size=region.sum())
        assert energy(pr, candidate) == pytest.approx(energy(sub, candidate[window]))
    inner = sub.active.copy()
    inner[0] = False
    nested, w2 = condition(sub, labels[window], inner)
    assert energy(nested, labels[window][w2]) == pytest.approx(energy(pr, labels))


@pytest.mark.parametrize("workers", [1, 2])
def test_sparse_icm_and_blocks_lower_energy(workers):
    pr, rng = problem()
    labels = rng.integers(5, size=pr.shape)
    icm = get_solver({"name": "icm", "max_sweeps": 100})
    result = icm.solve(pr, labels)
    assert np.all(np.diff(result.trace) <= 1e-8)
    costs = local_costs(pr, result.labels)
    current = np.take_along_axis(costs, result.labels[..., None], -1)[..., 0]
    assert np.all(costs.min(-1) >= current-1e-8)
    solver = get_solver({"name": "blocks", "tile": 4, "workers": workers,
                         "max_rounds": 2, "inner": {"name": "icm", "max_sweeps": 3}})
    result = solver.solve(pr, labels)
    assert np.all(np.diff([t["energy"] for t in result.trace]) <= 1e-8)
    assert result.energy == pytest.approx(energy(pr, result.labels))


def test_real_icm_toulbar_chain():
    pytest.importorskip("pytoulbar2")
    pr, _ = problem()
    region = np.zeros(pr.shape, bool)
    region[:2, :3] = True
    init = pr.unary.argmin(-1)
    pr, window = condition(pr, init, region)
    init = init[window]
    solver = get_solver({"name": "chain", "stages": [
        {"name": "icm"}, {"name": "toulbar2", "time_limit": 1}]})
    first = solver.stages[0].solve(pr, init)
    result = solver.solve(pr, init)
    assert result.energy <= first.energy+1e-8


def test_empty_prediction_shard_is_indexable(tmp_path):
    from wcsp.classifiers.cache import ProbStore
    store = ProbStore(tmp_path, "bcss", "empty", "dev")
    store.write(0, 1, [], [])
    assert store.ids() == []
