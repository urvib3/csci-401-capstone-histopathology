import csv

import numpy as np
import pytest

from experiments.bcss_wcsp.update_deliverable import BEGIN, END, replace_section, run_summary


def shard(directory, index, count, sample_id):
    name = directory / f'shard_{index:04d}_of_{count:04d}'
    cm = np.eye(5, dtype=np.int64)[None]
    np.savez(name.with_suffix('.npz'), ids=[sample_id], argmax_confusion=cm, wcsp_confusion=cm)
    with name.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'seconds', 'energy_argmax', 'energy_wcsp'])
        writer.writeheader()
        writer.writerow({'id':sample_id, 'seconds':2, 'energy_argmax':10, 'energy_wcsp':8})


def test_report_withholds_partial_shard_metrics(tmp_path):
    shard(tmp_path, 0, 2, 'a')
    partial = run_summary(tmp_path)
    assert not partial['complete']
    assert 'argmax' not in partial
    assert partial['samples'] == 1
    shard(tmp_path, 1, 2, 'b')
    complete = run_summary(tmp_path)
    assert complete['complete']
    assert complete['argmax']['mIoU'] == 1
    assert complete['energy_reduction_pct'] == pytest.approx(20)


def test_report_rejects_duplicate_sample_ids(tmp_path):
    shard(tmp_path, 0, 2, 'a')
    shard(tmp_path, 1, 2, 'a')
    with pytest.raises(ValueError, match='duplicate'):
        run_summary(tmp_path)


def test_refresh_preserves_other_deliverable_sections_and_is_repeatable():
    original='# Deliverable 6\nExisting context\n'
    first=replace_section(original, BEGIN+'\nfirst result\n'+END)
    first += '\n\n## Other results\nKeep these conclusions\n'
    second=replace_section(first, BEGIN+'\nnew result\n'+END)
    assert second.startswith('# Deliverable 6\nExisting context')
    assert second.endswith('Keep these conclusions\n')
    assert second.count(BEGIN) == second.count(END) == 1
    assert 'first result' not in second
    assert 'new result' in second
    with pytest.raises(ValueError, match='incomplete'):
        replace_section(original+BEGIN, 'new')


def test_refresher_refuses_to_overwrite_historical_deliverable_5(monkeypatch):
    from experiments.bcss_wcsp import update_deliverable as report
    monkeypatch.setattr('sys.argv', ['update_deliverable.py', '--deliverable',
                                    str(report.REPO/'deliverables/deliverable_5.md')])
    with pytest.raises(SystemExit) as error:
        report.main()
    assert error.value.code == 2


def test_default_refresh_creates_deliverable_6_without_touching_5(monkeypatch, tmp_path):
    from experiments.bcss_wcsp import update_deliverable as report
    d5=tmp_path/'deliverable_5.md'
    d5.write_text('Historical results must remain unchanged.\n')
    d6=tmp_path/'deliverable_6.md'
    monkeypatch.setattr(report, 'DEFAULT_DELIVERABLE', d6)
    monkeypatch.setattr('sys.argv', ['update_deliverable.py', '--output-root', str(tmp_path/'outputs'),
                                    '--training-root', str(tmp_path/'training')])
    report.main()
    assert d5.read_text() == 'Historical results must remain unchanged.\n'
    assert d6.read_text().startswith('# Deliverable 6')
    assert BEGIN in d6.read_text() and END in d6.read_text()
