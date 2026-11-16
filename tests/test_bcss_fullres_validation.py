import json

import numpy as np
import pytest

from experiments.bcss_wcsp.multiscale.fullres_validation import (
    cache_prediction, prepare, report, select_icm,
)
from experiments.bcss_wcsp.multiscale.fullres_sweep import sha256
from wcsp.classifiers.base import Prediction
from wcsp.data.base import Sample


def dev_rows():
    return [{'method': 'icm', 'mu': mu, 'samples': 185, 'expected': 185,
             'status': 'Complete', 'wcsp_mIoU': .65 + .001 * mu}
            for mu in (0, .05, .1, .2, .5, 1)]


def test_selection_requires_complete_dev_grid_and_uses_dev_only():
    rows = dev_rows()
    # A different family cannot silently change the predeclared ICM selection.
    rows.append({'method': 'hybrid', 'mu': .5, 'wcsp_mIoU': .99})
    assert select_icm(rows, 185)['mu'] == 1
    rows[0]['status'] = 'Incomplete'
    with pytest.raises(ValueError, match='complete'):
        select_icm(rows, 185)


def test_cached_baseline_uses_saved_probability_precision(tmp_path):
    probs = np.zeros((5, 4, 4), dtype=np.float32)
    probs[0], probs[1] = .49995, .50005
    assert np.all(probs.argmax(0) == 1)
    sample = Sample(id='tile', image=np.zeros((4, 4, 3), np.uint8),
                    target=np.zeros((4, 4), np.uint8), meta={'roi': 'roi'})
    record = cache_prediction(tmp_path, sample, Prediction(probs), 0, 'frozen')
    with np.load(tmp_path/'inputs/0000.npz') as data:
        assert np.all(data['probs'].argmax(0) == 0)
    assert record['argmax_confusion'][0][0] == 16


def test_wholly_ignored_tiles_are_recorded_without_inference(tmp_path):
    sample = Sample(id='ignored', image=np.zeros((4, 4, 3), np.uint8),
                    target=np.full((4, 4), 255, np.uint8), meta={'roi': 'roi'})
    record = cache_prediction(tmp_path, sample, None, 7, 'frozen')
    assert not record['scored'] and record['labeled_pixels'] == 0
    assert (tmp_path/'inputs/0007.json').exists()
    assert not (tmp_path/'inputs/0007.npz').exists()


def test_preparation_requires_full_coverage_including_ignored_tiles(tmp_path):
    selection = {'files': {}, 'validation_tiles': 2, 'dataset_ids': ['scored', 'ignored'],
                 'selected_mu': 1, 'treatments': [{'name': 'test', 'mu': 1, 'config': {}}]}
    path = tmp_path/'selection.json'
    path.write_text(json.dumps(selection))
    digest = sha256(path)
    target = (np.arange(16).reshape(4, 4) % 5).astype(np.uint8)
    probs = np.eye(5)[target].transpose(2, 0, 1).astype(np.float32)
    sample = Sample(id='scored', image=np.zeros((4, 4, 3), np.uint8), target=target,
                    meta={'roi': 'roi'})
    cache_prediction(tmp_path, sample, Prediction(probs), 0, digest)
    with pytest.raises(ValueError, match='Incomplete predictions'):
        prepare(tmp_path)
    sample = Sample(id='ignored', image=np.zeros((4, 4, 3), np.uint8),
                    target=np.full((4, 4), 255, np.uint8), meta={'roi': 'roi'})
    cache_prediction(tmp_path, sample, None, 1, digest)
    prepare(tmp_path)
    protocol = json.loads((tmp_path/'protocol.json').read_text())
    assert len(protocol['samples']) == 1
    assert protocol['ignored_tiles'] == ['ignored']
    assert protocol['baseline']['mIoU'] == 1


def test_validation_report_withholds_partial_result_metrics(tmp_path):
    protocol = {'selected_mu': 1, 'validation_tiles': 2, 'ignored_tiles': [],
                'baseline': {'mIoU': .5},
                'samples': [{'index': 0, 'id': 'a', 'roi': 'roi'},
                            {'index': 1, 'id': 'b', 'roi': 'roi'}],
                'treatments': [{'name': 'test', 'mu': 1}]}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    directory = tmp_path/'runs/test/val/samples'
    directory.mkdir(parents=True)
    (directory/'0000.json').write_text(json.dumps({'index': 0, 'id': 'a'}))
    (directory/'0000.npz').touch()
    report(tmp_path)
    row = json.loads((tmp_path/'summary.json').read_text())['results'][0]
    assert row['status'] == 'Incomplete' and row['samples'] == 1
    assert 'metrics' not in row and 'vs_argmax' not in row
