"""Frozen full-validation evaluation of the development-selected full-512 ICM model.

Stages: freeze (dev only), predict (GPU array), prepare, solve (CPU array), report.
The existing development runner and solver sources remain unchanged.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.bcss_wcsp.multiscale import fullres_sweep as dev
from wcsp import pipeline
from wcsp.config import ExperimentConfig, ProblemConfig
from wcsp.evaluate.compare import paired_bootstrap
from wcsp.evaluate.pixel import pixel_confusion, pixel_report
from wcsp.labels import IGNORE
from wcsp.model import build_problem
from wcsp.solve.energy import energy
from wcsp.solve.factory import get_solver

DEFAULT_ROOT = REPO / 'data/fullres_multiscale_val_20261115'


def select_icm(rows, expected):
    candidates = [r for r in rows if r['method'] == 'icm']
    if (len(candidates) != len(dev.MUS) or {r['mu'] for r in candidates} != set(dev.MUS)
            or any(r['status'] != 'Complete' or r['samples'] != expected
                   or r['expected'] != expected for r in candidates)):
        raise ValueError('The entire ICM development mu grid must be complete')
    return max(candidates, key=lambda r: (r['wcsp_mIoU'], -r['mu']))


def config(values):
    return ExperimentConfig(name=values['name'], dataset=values['dataset'],
                            classifier=values['classifier'],
                            problem=ProblemConfig.from_dict(values['problem']),
                            solver=values['solver'], paths=values.get('paths', {}))


def freeze(root, dev_root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'selection.json'
    if path.exists():
        verify(root)
        print(f'{path} already frozen; selection preserved.', flush=True)
        return
    source = dev.verify(dev_root)
    rows = json.loads((dev_root / 'summary.json').read_text())
    best = select_icm(rows, len(source['samples']))
    treatments = []
    for mu in (0, best['mu']):
        original = next(t for t in source['treatments'] if t['method'] == 'icm' and t['mu'] == mu)
        values = copy.deepcopy(original['config'])
        values['dataset']['split'] = 'val'
        values['name'] = f'full512_val_icm_mu{mu:g}'
        values['paths']['runs'] = str(root / 'runs')
        treatments.append({'name': values['name'], 'method': 'icm', 'mu': mu, 'config': values})
    if treatments[0]['mu'] == treatments[1]['mu']:
        treatments = treatments[:1]
    data = pipeline.dataset_for(config(treatments[0]['config']), 'val')
    if len(data) != 4028 or len(set(data.ids)) != len(data):
        raise ValueError('Expected all 4,028 unique preparation-v2 validation tiles')
    files = dict(source['files'])
    for p in [Path(__file__), REPO / 'wcsp/data/bcss.py', REPO / 'wcsp/labels.py',
              REPO / 'wcsp/pipeline.py', *sorted((REPO / 'wcsp/classifiers').glob('*.py'))]:
        files[str(p.resolve().relative_to(REPO))] = dev.sha256(p)
    selection = {'frozen_at': datetime.now(timezone.utc).isoformat(),
                 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 'split': 'val', 'selected_method': 'icm', 'selected_mu': best['mu'],
                 'selection_rule': 'Highest pooled dev mIoU over the complete six-value ICM grid; ties prefer lower mu.',
                 'dev_best': best, 'dev_summary_sha256': dev.sha256(dev_root / 'summary.json'),
                 'dev_protocol_sha256': dev.sha256(dev_root / 'protocol.json'),
                 'validation_tiles': len(data), 'dataset_ids': data.ids,
                 'treatments': treatments, 'files': files}
    dev.save_json(path, selection)
    print(f"Frozen ICM mu={best['mu']}; dev mIoU={best['wcsp_mIoU']:.6f}; "
          f'all {len(data)} validation tiles; {len(treatments)} fixed arms.', flush=True)


def verify(root):
    selection = json.loads((root / 'selection.json').read_text())
    for relative, digest in selection['files'].items():
        if dev.sha256(REPO / relative) != digest:
            raise ValueError(f'Frozen source/model changed: {relative}')
    return selection


def cache_prediction(root, sample, prediction, position, selection_hash):
    directory = root / 'inputs'
    directory.mkdir(parents=True, exist_ok=True)
    target = sample.grid_target
    record = {'index': position, 'id': sample.id, **sample.meta,
              'labeled_pixels': int((target != IGNORE).sum()), 'selection_sha256': selection_hash}
    record['scored'] = record['labeled_pixels'] > 0
    if record['scored']:
        probs = prediction.probs.astype(np.float16)
        # Score exactly the same stored precision and tie behavior the WCSP reads.
        record['argmax_confusion'] = pixel_confusion(probs.argmax(0), target).tolist()
        path = directory / f'{position:04d}.npz'
        temporary = path.with_name(f'{position:04d}.{os.getpid()}.partial.npz')
        np.savez(temporary, id=sample.id, probs=probs, target=target)
        temporary.replace(path)
        record['sha256'] = dev.sha256(path)
    dev.save_json(directory / f'{position:04d}.json', record)
    return record


def predict(root, shard, shards, batch_size):
    selection = verify(root)
    selection_hash = dev.sha256(root / 'selection.json')
    cfg = config(copy.deepcopy(selection['treatments'][0]['config']))
    cfg.classifier.update(device='cuda', threads=4)
    data = pipeline.dataset_for(cfg, 'val')
    if data.ids != selection['dataset_ids']:
        raise ValueError('Validation population changed after freezing')
    directory = root / 'inputs'
    directory.mkdir(exist_ok=True)
    positions = []
    for position in range(shard, len(data), shards):
        meta = directory / f'{position:04d}.json'
        if meta.exists():
            r = json.loads(meta.read_text())
            if r['selection_sha256'] != selection_hash or r['id'] != data.ids[position]:
                raise ValueError('Existing prediction belongs to another selection')
            if not r['scored'] or (directory / f'{position:04d}.npz').exists():
                continue
        positions.append(position)
    classifier = None
    for begin in range(0, len(positions), batch_size):
        scored = []
        for position in positions[begin:begin + batch_size]:
            sample = data[position]
            if pipeline._scored(sample):
                scored.append((position, sample))
            else:
                cache_prediction(root, sample, None, position, selection_hash)
        if scored:
            if classifier is None:
                classifier = pipeline.classifier_for(cfg)
            predictions = classifier.predict([sample for _, sample in scored])
            for (position, sample), prediction in zip(scored, predictions):
                probs = prediction.probs
                if (probs.shape != (5, 512, 512) or not np.isfinite(probs).all()
                        or not np.allclose(probs.sum(0), 1, atol=.002)):
                    raise ValueError(f'Invalid full-resolution prediction: {sample.id}')
                cache_prediction(root, sample, prediction, position, selection_hash)
        print(f'Prediction shard {shard}/{shards}: '
              f'{min(begin + batch_size, len(positions))}/{len(positions)} new tiles', flush=True)


def prepare(root):
    selection = verify(root)
    destination = root / 'protocol.json'
    if destination.exists():
        print('Validation inputs already prepared; preserving protocol.', flush=True)
        return
    selection_hash = dev.sha256(root / 'selection.json')
    paths = list((root / 'inputs').glob('*.json'))
    if len(paths) != selection['validation_tiles']:
        raise ValueError(f'Incomplete predictions: {len(paths)}/{selection["validation_tiles"]}')
    records, seen, ignored = [], set(), []
    baseline = np.zeros((5, 5), dtype=np.int64)
    for path in sorted(paths):
        record = json.loads(path.read_text())
        position = record['index']
        if (position in seen or not 0 <= position < selection['validation_tiles']
                or record['id'] != selection['dataset_ids'][position]
                or record['selection_sha256'] != selection_hash):
            raise ValueError(f'Wrong/duplicate prediction record: {path}')
        seen.add(position)
        if not record['scored']:
            ignored.append(record['id'])
            continue
        source = path.with_suffix('.npz')
        if dev.sha256(source) != record['sha256']:
            raise ValueError(f'Prediction checksum mismatch: {source}')
        baseline += np.asarray(record['argmax_confusion'], dtype=np.int64)
        records.append(record)
    if len(seen) != selection['validation_tiles']:
        raise ValueError('Validation prediction coverage is incomplete')
    protocol = {**selection, 'selection_sha256': selection_hash, 'samples': records,
                'ignored_tiles': ignored, 'baseline': pixel_report(baseline)}
    for treatment in protocol['treatments']:
        directory = root / 'runs' / treatment['name'] / 'val'
        (directory / 'samples').mkdir(parents=True, exist_ok=True)
        dev.save_json(directory / 'config.json', treatment['config'])
    dev.save_json(destination, protocol)
    print(f'All {len(seen)} validation tiles processed; {len(records)} scored, '
          f'{len(ignored)} wholly ignored; raw U-Net mIoU={protocol["baseline"]["mIoU"]:.6f}', flush=True)


def solve(root, treatment_index, shard, shards):
    selection = verify(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['selection_sha256'] != dev.sha256(root / 'selection.json'):
        raise ValueError('Selection changed after preparing inputs')
    treatment = selection['treatments'][treatment_index]
    directory = root / 'runs' / treatment['name'] / 'val' / 'samples'
    solver = get_solver(treatment['config']['solver'])
    problem_cfg = ProblemConfig.from_dict(treatment['config']['problem'])
    for record in protocol['samples'][shard::shards]:
        stem = f"{record['index']:04d}"
        result_json = directory / f'{stem}.json'
        if result_json.exists() and result_json.with_suffix('.npz').exists():
            continue
        source = root / 'inputs' / f'{stem}.npz'
        if dev.sha256(source) != record['sha256']:
            raise ValueError(f'Prediction checksum mismatch: {source}')
        with np.load(source) as z:
            probs, target = z['probs'].astype(np.float32), z['target']
        problem = build_problem(probs, problem_cfg)
        initial = problem.unary.argmin(-1)
        print(f"START {treatment['name']} {stem} {record['id']}", flush=True)
        result = solver.solve(problem, initial)
        energies = [t['energy'] for t in result.trace]
        if np.any(np.diff(energies) > 1e-6):
            raise ValueError('Hierarchy energy increased')
        arrays = {'id': record['id'], 'argmax': initial.astype(np.uint8),
                  'wcsp': result.labels.astype(np.uint8),
                  'argmax_confusion': pixel_confusion(initial, target),
                  'wcsp_confusion': pixel_confusion(result.labels, target)}
        arrays.update({f'level_{i+1}': z.astype(np.uint8) for i, z in enumerate(result.levels)})
        temporary = directory / f'{stem}.{os.getpid()}.partial.npz'
        np.savez_compressed(temporary, **arrays)
        temporary.replace(directory / f'{stem}.npz')
        dev.save_json(result_json, {'id': record['id'], 'index': record['index'],
                      'roi': record['roi'], 'seconds': result.seconds,
                      'changed': float((initial != result.labels).mean()),
                      'pixel_energy_before': energy(problem, initial),
                      'pixel_energy_after': energy(problem, result.labels),
                      'hierarchy_energy_before': energies[0], 'hierarchy_energy_after': result.energy,
                      'trace': result.trace})
        print(f'DONE {stem} {result.seconds:.1f}s', flush=True)


def report(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    rows, scores, baseline_arrays = [], {}, None
    groups = [r['roi'] for r in protocol['samples']]
    for treatment in protocol['treatments']:
        directory = root / 'runs' / treatment['name'] / 'val' / 'samples'
        found, complete = dev.completed_samples(directory, protocol['samples'])
        row = {'method': 'icm', 'mu': treatment['mu'], 'samples': len(found),
               'expected': len(protocol['samples']), 'status': 'Complete' if complete else 'Incomplete'}
        if complete:
            before, after = [], []
            for index in sorted(found):
                with np.load(directory / f'{index:04d}.npz') as z:
                    if str(z['id']) != found[index]['id']:
                        raise ValueError('Result array/metadata ID mismatch')
                    before.append(z['argmax_confusion'])
                    after.append(z['wcsp_confusion'])
            before, after = np.asarray(before), np.asarray(after)
            if not np.isclose(pixel_report(before.sum(0))['mIoU'], protocol['baseline']['mIoU'], atol=1e-12):
                raise ValueError('Result argmax differs from frozen input baseline')
            baseline_arrays = before
            scores[treatment['mu']] = after
            row.update(metrics=pixel_report(after.sum(0)),
                       mean_seconds=float(np.mean([r['seconds'] for r in found.values()])),
                       vs_argmax=paired_bootstrap(before, after, groups))
        rows.append(row)
    for row in rows:
        if row['mu'] in scores and 0 in scores:
            row['vs_mu0'] = paired_bootstrap(scores[0], scores[row['mu']], groups)
    summary = {'split': 'val', 'selected_mu': protocol['selected_mu'],
               'total_validation_tiles': protocol['validation_tiles'],
               'scored_tiles': len(protocol['samples']), 'ignored_tiles': len(protocol['ignored_tiles']),
               'raw_unet': protocol['baseline'], 'results': rows}
    dev.save_json(root / 'summary.json', summary)
    lines = ['# Full-resolution BCSS validation', '',
             f"Frozen dev-selected ICM mu={protocol['selected_mu']}; "
             'no parameters selected on validation.', '',
             f"All {protocol['validation_tiles']} tiles processed at 512x512; "
             f"{len(protocol['samples'])} scored, {len(protocol['ignored_tiles'])} wholly ignored.", '',
             f"Raw U-Net mIoU: **{protocol['baseline']['mIoU']:.6f}**.", '',
             '| Method | mu | Finished / expected | mIoU | Mean s/tile | Status |',
             '|---|---:|---:|---:|---:|---|']
    for row in rows:
        metric = f"{row['metrics']['mIoU']:.6f}" if 'metrics' in row else '—'
        seconds = f"{row['mean_seconds']:.1f}" if 'mean_seconds' in row else '—'
        lines.append(f"| ICM | {row['mu']} | {row['samples']} / {row['expected']} | "
                     f"{metric} | {seconds} | {row['status']} |")
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'predict', 'prepare', 'solve', 'report'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--dev-root', type=Path, default=dev.DEFAULT_ROOT)
    parser.add_argument('--treatment', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.batch_size < 1:
        parser.error('invalid shard or batch size')
    os.chdir(REPO)
    root = args.root.resolve()
    if args.command == 'freeze':
        freeze(root, args.dev_root.resolve())
    elif args.command == 'predict':
        predict(root, args.shard, args.shards, args.batch_size)
    elif args.command == 'prepare':
        prepare(root)
    elif args.command == 'solve':
        solve(root, args.treatment, args.shard, args.shards)
    else:
        report(root)


if __name__ == '__main__':
    main()
