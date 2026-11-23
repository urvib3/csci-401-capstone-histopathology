"""Verify isolation, fit-only priors, selection, saved metrics and paired inference."""
import ast
import hashlib
import json
import io
import zipfile
from pathlib import Path

import numpy as np

from families import HERE, metric, read, select, write
from report import critical_probability, holm, scores


def main():
    split = read(HERE / 'split.json')
    groups = {}
    for group in ['fit', 'dev', 'test']:
        groups[group] = {row['image_index'] for row in split[group]}
    assert not (groups['fit'] & groups['dev'] or groups['fit'] & groups['test'] or groups['dev'] & groups['test'])
    assert not set.union(*groups.values()) & set(split['excluded_ids'])
    assert len(split['excluded_ids']) == 1235
    assert len(groups['test']) == 169
    tables = read(HERE / 'priors_and_tables.json')
    assert set(tables['fit_patch_ids']) == groups['fit']
    set_counts = np.array(tables['type_set']['counts'])
    assert set_counts.sum() == 168
    assert np.all(set_counts.sum(1) >= 10)
    frequency = np.ones(5)
    observed_sets = np.zeros_like(set_counts)
    for index in tables['fit_patch_ids']:
        with np.load(HERE / 'cache' / f'gt_{index:04d}.npz') as archive:
            kinds = archive['types'][1:] - 1
        np.add.at(frequency, kinds, 1)
        mask = int(np.bitwise_or.reduce(1 << kinds.astype(int), initial=0))
        band = np.searchsorted(tables['type_set']['boundaries'], len(kinds), side='right')
        observed_sets[band, mask] += 1
    np.testing.assert_array_equal(observed_sets, set_counts)
    np.testing.assert_allclose(frequency / frequency.sum(), tables['type_frequency'])
    archive_path = HERE.parents[3] / 'benchmarks/pannuke/data/fold_2.zip'
    with zipfile.ZipFile(archive_path) as archive:
        tissue_files = [name for name in archive.namelist() if name.endswith('types.npy')]
        tissue = np.load(io.BytesIO(archive.read(tissue_files[0])))
    eligible = [int(index) for index in np.flatnonzero(tissue == 'Breast') if int(index) not in split['excluded_ids']]
    np.random.default_rng(20261122).shuffle(eligible)
    fit_end = int(.4 * len(eligible))
    dev_end = fit_end + int(.2 * len(eligible))
    expected = {'fit': eligible[:fit_end], 'dev': eligible[fit_end:dev_end], 'test': eligible[dev_end:]}
    for group in expected:
        assert expected[group] == [row['image_index'] for row in split[group]], group
    frozen = read(HERE / 'frozen.json')
    development = read(HERE / 'development.json')
    assert len(frozen['families']) == 8
    for family, winner in frozen['families'].items():
        rows = development[family]
        if family == 'type_set':
            maximum = max(row['bio_miou'] for row in rows)
            close = [row for row in rows if row['bio_miou'] >= maximum - .0001]
            chosen = min(close, key=lambda row: (row['config']['strength'],
                row['config']['mode'] != 'presence', ['vote', 'mean', 'log_mean'].index(row['config']['unary'])))
        else:
            chosen = select(rows)
        assert chosen['config'] == winner, family
    protocol = read(HERE / 'expansion_protocol.json')
    indices = sorted(np.random.default_rng(20261122).choice(len(split['dev']), 32, replace=False).tolist())
    assert protocol['dev_pixel_ids'] == [split['dev'][index]['image_index'] for index in indices]
    assert set(protocol['dev_pixel_ids']) <= groups['dev']
    receipt = read(HERE / 'test_receipt.json')
    assert receipt['frozen_sha256'] == hashlib.sha256((HERE / 'frozen.json').read_bytes()).hexdigest()
    with np.load(HERE / 'all_confusions.npz') as archive:
        matrices = {name: archive[name] for name in archive.files}
    with np.load(HERE / 'pilot_confusions.npz') as archive:
        for name in archive.files:
            np.testing.assert_array_equal(archive[name], matrices[name])
    evaluation = read(HERE / 'evaluation.json')
    count = len(groups['test'])
    draws = np.random.default_rng(20261122).integers(count, size=(5000, count))
    weights = np.zeros((5000, count), int)
    np.add.at(weights, (np.arange(5000)[:, None], draws), 1)
    samples = {}
    for name, matrix in matrices.items():
        assert matrix.shape == (count, 6, 6)
        np.testing.assert_array_equal(matrix.sum((1, 2)), 256 * 256)
        np.testing.assert_array_equal(matrix.sum(2), matrices['raw_tp'].sum(2))
        for key, value in metric(matrix).items():
            np.testing.assert_allclose(value, evaluation['methods'][name][key])
        samples[name] = scores(matrix, weights)
    pvalues = {}
    for family, result in evaluation['results'].items():
        for label, comparison in result['comparisons'].items():
            differences = samples['breast_' + family] - samples[comparison['method']]
            np.testing.assert_allclose(np.quantile(differences, [.025, .975]), comparison['ci95'])
            assert comparison['wins'] + comparison['losses'] + comparison['ties'] == count
            if label == 'matched_control':
                tail = min(int((differences <= 0).sum()), int((differences >= 0).sum()))
                pvalues[family] = min(1., 2 * (tail + 1) / 5001)
    adjusted = holm(pvalues)
    for family, result in evaluation['results'].items():
        comparison = result['comparisons']['matched_control']
        np.testing.assert_allclose(comparison['p_holm'], adjusted[family])
        critical = critical_probability(family, pvalues)
        tail = max(0., (critical / 2 - 1 / 5001) * 5001 / 5000)
        difference = samples['breast_' + family] - samples[comparison['method']]
        interval = np.quantile(difference, [tail, 1 - tail])
        np.testing.assert_allclose(interval, comparison['holm_ci95'])
        assert result['works'] == bool(interval[0] > 0 and adjusted[family] < .05)
    root = HERE.parents[3]
    for filename, digest in read(HERE / 'protocol.json')['sources'].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest
    for path in HERE.glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert not node.id.startswith('_') or node.id in {'__name__', '__file__', '__doc__'}, (path, node.id)
            if isinstance(node, ast.arg):
                assert not node.arg.startswith('_'), (path, node.arg)
    for group in ['dev', 'test']:
        inference = read(HERE / f'inference_{group}.json')
        assert inference['patch_ids'] == [row['image_index'] for row in split[group]]
        assert inference['checkpoint_sha256'] == '789b2be8bdaf532627f584a009f83b3ba06e04c0e55c1635252ff12785ef57e8'
    write('validation.json', {'passed': True, 'counts': {name: len(values) for name, values in groups.items()},
          'all_eight_selections_checked': True, 'pilot_confusions_unchanged': True,
          'fit_only_prior_reconstructed': True, 'breast_archive_split_reconstructed': True, 'all_scores_and_intervals_reproduced': True,
          'test_receipt_matches_frozen': True, 'original_objective_sources_unchanged': True,
          'code_variable_naming_checked': True})
    print('Verified all eight frozen selections, fit-only priors, split isolation, metrics, paired intervals and Holm decisions.')


if __name__ == '__main__':
    main()
