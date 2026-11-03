import json

import numpy as np
import pandas as pd
import pytest

from wcsp.config import ExperimentConfig, ProblemConfig
from wcsp.model.problem import build_problem
from wcsp.solve.factory import get_solver
from wcsp.solve.icm import ICMSolver
from wcsp.solve.instrumentation import capture_events, iteration_totals


def toy(shape=(9, 9), lam=8):
    rng = np.random.default_rng(19)
    probs = rng.dirichlet(np.ones(5), size=shape).transpose(2, 0, 1)
    return build_problem(probs, ProblemConfig(lam=lam)), rng.integers(5, size=shape), probs


def test_icm_counts_distinguish_convergence_and_sweep_cap():
    pr, labels, _ = toy()
    limited = ICMSolver(max_sweeps=1).solve(pr, labels)
    assert limited.stats['sweeps'] == 1
    assert limited.stats['stop_reason'] == 'sweep_limit'
    assert not limited.stats['converged']
    assert limited.stats['energies'] == limited.trace
    pr, _, _ = toy(lam=0)
    stable = ICMSolver().solve(pr, pr.unary.argmin(-1))
    assert stable.stats['sweeps'] == 1
    assert stable.stats['converged']
    assert stable.stats['pixel_updates_per_sweep'] == [0]


def test_nested_stage_and_block_counts_match_streamed_events():
    pr, labels, _ = toy()
    solver = get_solver({'name': 'blocks', 'block': 3, 'max_rounds': 2, 'inner': {
        'name': 'chain', 'stages': [{'name': 'icm', 'max_sweeps': 3}, {'name': 'icm', 'max_sweeps': 2}]}})
    events = []
    with capture_events(events.append):
        result = solver.solve(pr, labels)
    counts = iteration_totals(result.stats)
    assert counts['icm_calls'] == 2 * len(result.stats['blocks'])
    sweeps = [e for e in events if e['event'] == 'icm_sweep']
    assert counts['icm_sweeps'] == len(sweeps)
    assert all('stage' in e and 'round' in e and 'window' in e for e in sweeps)
    assert counts['icm_pixel_updates'] == sum(e['changed_pixels'] for e in sweeps)
    assert result.stats['stop_reason'] in ('round_limit', 'energy_plateau')


def test_real_toulbar_counts_and_no_search_limit():
    pytest.importorskip('pytoulbar2')
    pr, init, _ = toy((1, 3))
    events = []
    with capture_events(events.append):
        result = get_solver({'name': 'toulbar2', 'time_limit': None}).solve(pr, init)
    assert result.stats['search_limit_seconds'] == 0
    assert result.stats['search_nodes'] >= 0
    assert result.stats['backtracks'] >= 0
    assert [e['event'] for e in events] == ['toulbar_search_start', 'toulbar_search_end']


def test_pipeline_persists_details_and_csv_totals(monkeypatch, tmp_path):
    from wcsp import pipeline
    from wcsp.classifiers.base import Prediction
    from wcsp.data.base import Sample
    _, target, probs = toy()
    sample = Sample('tile', np.zeros((9, 9, 3), dtype=np.uint8), target)
    class Data:
        ids = ['tile']
        def __len__(self): return 1
        def __getitem__(self, index): return sample
    class Store:
        def __contains__(self, key): return True
        def load(self, key): return Prediction(probs)
    monkeypatch.setattr(pipeline, 'dataset_for', lambda *args: Data())
    monkeypatch.setattr(pipeline, 'store_for', lambda *args: Store())
    cfg = ExperimentConfig('test', {'name': 'bcss', 'split': 'dev'},
        {'name': 'fake', 'temperature': 1}, ProblemConfig(),
        {'name': 'blocks', 'whole': True, 'max_rounds': 2, 'inner': {'name': 'icm', 'max_sweeps': 3}})
    path = pipeline.solve(cfg, 'dev', run_dir=tmp_path, save_labels=False)
    events = [json.loads(line) for line in path.with_suffix('.iterations.jsonl').read_text().splitlines()]
    assert len({e['attempt_id'] for e in events}) == 1
    assert all('timestamp' in e for e in events)
    assert any(e['event'] == 'icm_sweep' for e in events)
    last = events[-1]
    assert last['event'] == 'sample_end'
    row = pd.read_csv(path.with_suffix('.csv')).iloc[0]
    assert row['icm_sweeps'] == last['icm_sweeps']
    assert row['instrumentation_version'] == 1
    assert row['stop_reason'] == last['stats']['stop_reason']
