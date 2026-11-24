import copy
import json

import numpy as np
import pytest

from experiments.bcss_wcsp.multiscale.convergence_solver import MeasuredBlocks, MeasuredHierarchy, MeasuredICM
from experiments.bcss_wcsp.multiscale.fullres_expanded import aggregate, digest, solve, treatments
from wcsp.config import ProblemConfig
from wcsp.model import build_problem
from wcsp.solve import BlockCoordinateDescent, BlockLayout, ICMSolver, energy
from wcsp.solve.hierarchy import HierarchySolver
from wcsp.solve.instrumentation import capture_events, iteration_totals


def problem(seed=9):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(5), size=(16, 16)).transpose(2, 0, 1)
    return build_problem(probs, ProblemConfig(lam=12)), rng.integers(0, 5, (16, 16))


def test_finite_instrumented_descent_reproduces_legacy_labels_and_energy():
    pr, initial = problem()
    for mu in (0, 1):
        legacy = HierarchySolver(
            BlockCoordinateDescent(ICMSolver(max_sweeps=3),
                                   BlockLayout.cross(pr.shape, block=5), max_rounds=2),
            levels=2, mu=mu, max_rounds=4)
        measured = MeasuredHierarchy(MeasuredBlocks(MeasuredICM(3), block=5, max_rounds=2),
                                     levels=2, mu=mu, max_rounds=4, block_max_sweeps=50)
        before, after = legacy.solve(pr, initial), measured.solve(pr, initial)
        np.testing.assert_array_equal(before.labels, after.labels)
        for a, b in zip(before.levels, after.levels):
            np.testing.assert_array_equal(a, b)
        assert after.energy == pytest.approx(before.energy)
        np.testing.assert_allclose([r['energy'] for r in before.trace],
                                   [r['energy'] for r in after.trace], atol=1e-9)


def test_uncapped_descent_has_no_improving_pixel_or_coarse_label_move():
    pr, initial = problem(12)
    solver = MeasuredHierarchy(MeasuredBlocks(MeasuredICM(), block=5), levels=2, mu=1)
    result = solver.solve(pr, initial)
    assert result.stats['converged']
    assert result.stats['stop_reason'] == 'energy_plateau'
    assert result.stats['audit']['improving_variables'] == 0
    assert np.all(np.diff([r['energy'] for r in result.trace]) <= 1e-6)
    blocks = solver.block_problems(pr)
    assert result.energy == pytest.approx(solver.total_energy(pr, blocks, result.labels, result.levels))
    for level in range(3):
        current = result.labels if level == 0 else result.levels[level-1]
        # Independent objective check, rather than reusing audit conditional costs.
        for index in list(np.ndindex(current.shape))[:8]:
            for label in range(5):
                pixels = result.labels.copy()
                coarse = [z.copy() for z in result.levels]
                (pixels if level == 0 else coarse[level-1])[index] = label
                assert solver.total_energy(pr, blocks, pixels, coarse) >= result.energy - 1e-9


def test_all_nested_icm_sweeps_are_streamed_and_counted_once():
    pr, initial = problem(17)
    solver = MeasuredHierarchy(MeasuredBlocks(MeasuredICM(2), block=5, max_rounds=2),
                               levels=2, mu=1, max_rounds=2, block_max_sweeps=2)
    events = []
    with capture_events(events.append):
        result = solver.solve(pr, initial)
    totals = iteration_totals(result.stats)
    sweeps = [e for e in events if e['event'] == 'icm_sweep']
    assert totals['icm_sweeps'] == len(sweeps)
    assert totals['icm_pixel_updates'] == sum(e['changed_pixels'] for e in sweeps)
    assert totals['icm_calls'] == sum(e['event'] == 'icm_end' for e in events)
    assert all('hierarchy_round' in e and 'level' in e for e in sweeps)
    assert all('window' in e and 'round' in e for e in sweeps if e['level'] == 0)
    assert {e['level'] for e in sweeps} == {0, 1, 2}
    # Actual experiment sinks write JSON rather than retaining Python objects.
    json.dumps(events, allow_nan=False)
    json.dumps(result.stats, allow_nan=False)


def test_icm_cap_is_distinguished_from_no_changes():
    pr, initial = problem()
    limited = MeasuredICM(1).solve(pr, initial)
    assert limited.stats['sweeps'] == 1
    assert limited.stats['stop_reason'] == 'sweep_limit'
    assert not limited.stats['converged']
    stable = MeasuredICM().solve(pr, MeasuredICM().solve(pr, initial).labels)
    assert stable.stats['sweeps'] == 1
    assert stable.stats['stop_reason'] == 'no_changes'


def test_grid_removes_all_caps_and_preserves_matched_scale_weights():
    base = {'problem': {'lam': 8}, 'solver': {'mu': 1, 'block_lam': 8,
            'max_rounds': 4, 'pixel': {'max_rounds': 4, 'inner': {'max_sweeps': 50}}}}
    original = copy.deepcopy(base)
    grid = treatments(base)
    assert len(grid) == 17 and len({r['name'] for r in grid}) == 17
    for row in grid:
        solver = row['config']['solver']
        assert solver['max_rounds'] is None
        if row['mode'] == 'full':
            assert row['config']['problem']['lam'] == solver['block_lam'] == row['lam']
            assert solver['pixel']['max_rounds'] is None
            assert solver['pixel']['inner']['max_sweeps'] is None
            assert solver['block_max_sweeps'] is None
        else:
            assert solver['pixel']['max_rounds'] == 4
            assert solver['pixel']['inner']['max_sweeps'] == 50
    assert base == original


def test_complete_report_rejects_uncertified_convergence(tmp_path):
    sample = {'id': 'a', 'index': 0}
    (tmp_path/'0000.json').write_text(json.dumps(dict(**sample, converged=False,
        audit={'improving_variables': 1})))
    (tmp_path/'0000.npz').touch()
    with pytest.raises(ValueError, match='unaudited'):
        aggregate(tmp_path, [sample], True, .5)


def test_runner_persists_actual_sweep_counts_and_convergence_audit(tmp_path):
    source = tmp_path/'source'
    (source/'inputs').mkdir(parents=True)
    (source/'protocol.json').write_text('{}')
    rng = np.random.default_rng(43)
    probs = rng.dirichlet(np.ones(5), size=(16, 16)).transpose(2, 0, 1)
    path = source/'inputs/0000.npz'
    np.savez(path, id='tile', probs=probs.astype(np.float16), target=rng.integers(0, 5, (16, 16)))
    spec = dict(name='hierarchy', levels=2, size=4, mu=1, block_alpha=1, block_lam=8,
                block_types=[1, 2], max_rounds=None, block_max_sweeps=None,
                pixel=dict(name='blocks', block=5, max_rounds=None,
                           inner=dict(name='icm', max_sweeps=None)))
    record = dict(id='tile', index=0, roi='roi', sha256=digest(path))
    root = tmp_path/'output'
    root.mkdir()
    protocol = dict(source_root=str(source), source_protocol_sha256=digest(source/'protocol.json'),
                    files={}, samples=[record], treatments=[dict(name='test', mode='full', mu=1, lam=8,
                    config=dict(problem=ProblemConfig(lam=8).to_dict(), solver=spec))])
    (root/'protocol.json').write_text(json.dumps(protocol))
    solve(root, 0, 0, 1)
    result_path = root/'runs/test/dev/samples/0000.json'
    result = json.loads(result_path.read_text())
    events = [json.loads(line) for line in result_path.with_suffix('.iterations.jsonl').read_text().splitlines()]
    assert result['converged'] and result['audit']['improving_variables'] == 0
    assert result['icm_sweeps'] == sum(e['event'] == 'icm_sweep' for e in events)
    assert result['icm_calls'] == sum(e['event'] == 'icm_end' for e in events)
    assert result['icm_sweep_limit_hits'] == 0
    assert events[-1]['event'] == 'sample_end'
    with np.load(result_path.with_suffix('.npz')) as z:
        assert z['wcsp'].shape == (16, 16) and z['level_2'].shape == (2, 2)
