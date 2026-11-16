"""Freeze, run and report the paired full-512 multiscale mu sweep.

Outputs checkpoint each tile so a timeout does not discard earlier tiles.
Only complete, identical sample sets receive aggregate metrics in the report.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from wcsp import pipeline
from wcsp.config import ProblemConfig, load_config
from wcsp.data import select
from wcsp.evaluate.compare import paired_bootstrap
from wcsp.evaluate.pixel import pixel_confusion, pixel_report
from wcsp.model import build_problem
from wcsp.solve.energy import energy
from wcsp.solve.factory import get_solver

CONFIG = REPO / 'experiments/bcss_wcsp/configs/bcss_multiscale_full512.yaml'
DEFAULT_ROOT = REPO / 'data/fullres_multiscale_20261115'
MUS = (0, 0.05, 0.1, 0.2, 0.5, 1)
METHODS = ('icm', 'toulbar2', 'hybrid')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def treatments(base):
    result = []
    for method in METHODS:
        for mu in MUS:
            config = copy.deepcopy(base)
            config['solver']['mu'] = mu
            inner = {'name': 'icm', 'max_sweeps': 50}
            if method == 'toulbar2':
                inner = {'name': 'toulbar2', 'time_limit': 2, 'resolution': 3}
            elif method == 'hybrid':
                inner = {'name': 'chain', 'stages': [inner,
                         {'name': 'toulbar2', 'time_limit': 2, 'resolution': 3}]}
            config['solver']['pixel']['inner'] = inner
            result.append({'name': f'full512_ms_{method}_mu{mu:g}',
                           'method': method, 'mu': mu, 'config': config})
    return result


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'protocol.json'
    if manifest_path.exists():
        verify(root)
        print(f'{manifest_path} already frozen; inputs verified.', flush=True)
        return
    os.environ.setdefault('WCSP_DATA', str(REPO / 'data'))
    os.environ.setdefault('WCSP_FULLRES_CACHE', str(REPO / 'data/tonight/cache'))
    os.environ['WCSP_FULLRES_OUT'] = str(root)
    cfg = load_config(CONFIG)
    data = pipeline.dataset_for(cfg, 'dev')
    store = pipeline.store_for(cfg, 'dev')
    preparation = json.loads((Path(cfg.dataset['root']) / 'manifest.json').read_text())
    if preparation['preparation_version'] != 2 or len(data) != 1012:
        raise ValueError('Expected the preparation-v2 dataset with 1,012 dev tiles')
    cached = store.ids()
    if len(cached) != 185 or len(set(cached)) != 185:
        raise ValueError('Expected the 185 cached full-field dev samples')
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=True)
    records, baseline = [], np.zeros((5, 5), dtype=np.int64)
    for position in select(len(data), limit=200):
        sample = data[position]
        if not pipeline._scored(sample):
            if sample.id in cached:
                raise ValueError('Unexpected unscored sample in cache')
            continue
        if sample.id not in store:
            raise ValueError(f'Missing cached prediction: {sample.id}')
        prediction = store.load(sample.id)
        probs = prediction.probs
        if probs.shape != (5, 512, 512) or not np.isfinite(probs).all():
            raise ValueError(f'Invalid full-resolution probabilities: {sample.id}')
        if not np.allclose(probs.sum(0), 1, atol=0.002):
            raise ValueError(f'Probabilities are not normalized: {sample.id}')
        index = len(records)
        path = inputs / f'{index:04d}.npz'
        np.savez(path, probs=probs.astype(np.float16), target=sample.grid_target,
                 id=sample.id)
        baseline += pixel_confusion(probs.argmax(0), sample.grid_target)
        records.append({'index': index, 'position': position, 'id': sample.id,
                        **sample.meta, 'sha256': sha256(path)})
    if set(cached) != {r['id'] for r in records}:
        raise ValueError('Cached population differs from the frozen dev selection')
    paths = [CONFIG, Path(cfg.classifier['checkpoint']), Path(cfg.problem.table),
             Path(cfg.dataset['root']) / 'manifest.json']
    paths += sorted((REPO / 'wcsp/solve').glob('*.py'))
    paths += [REPO / 'wcsp/model/problem.py', REPO / 'wcsp/model/unary.py', Path(__file__)]
    files = {str(p.resolve().relative_to(REPO)): sha256(p) for p in paths}
    protocol = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'split': 'dev', 'limit_before_ignored_filter': 200, 'samples': records,
                'baseline': pixel_report(baseline), 'files': files,
                'cache': str(store.dir),
                'cache_files': {str(p.relative_to(REPO)): sha256(p)
                                for p in sorted(store.dir.glob('shard_*_of_*.npz'))},
                'treatments': treatments(cfg.to_dict())}
    for treatment in protocol['treatments']:
        directory = root / 'runs' / treatment['name'] / 'dev'
        (directory / 'samples').mkdir(parents=True, exist_ok=True)
        save_json(directory / 'config.json', treatment['config'])
    save_json(manifest_path, protocol)
    print(f'Frozen {len(records)} full-512 dev tiles; baseline mIoU '
          f"{protocol['baseline']['mIoU']:.6f}; {len(protocol['treatments'])} treatments", flush=True)


def verify(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    for relative, digest in protocol['files'].items():
        if sha256(REPO / relative) != digest:
            raise ValueError(f'Frozen source/input changed: {relative}; use a new output root')
    return protocol


def solve(root, treatment_index, shard, shards, smoke=False):
    protocol = verify(root)
    treatment = protocol['treatments'][treatment_index]
    directory = root / ('smoke' if smoke else 'runs') / treatment['name'] / 'dev' / 'samples'
    directory.mkdir(parents=True, exist_ok=True)
    solver = get_solver(treatment['config']['solver'])
    problem_cfg = ProblemConfig.from_dict(treatment['config']['problem'])
    records = protocol['samples'][shard::shards]
    if smoke:
        # A tile with substantial labeled tissue, not an empty/all-one-class shortcut.
        records = [protocol['samples'][0]]
    for record in records:
        stem = f"{record['index']:04d}"
        result_json = directory / f'{stem}.json'
        if result_json.exists() and (directory / f'{stem}.npz').exists():
            continue
        path = root / 'inputs' / f'{stem}.npz'
        if sha256(path) != record['sha256']:
            raise ValueError(f'Frozen probability/target input changed: {path}')
        with np.load(path) as z:
            probs, target = z['probs'].astype(np.float32), z['target']
        problem = build_problem(probs, problem_cfg)
        initial = problem.unary.argmin(-1)
        print(f"START {treatment['name']} {record['index']} {record['id']}", flush=True)
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
        # Pixel and augmented hierarchy objectives are deliberately separate.
        save_json(result_json, {'id': record['id'], 'index': record['index'],
                  'roi': record['roi'], 'seconds': result.seconds,
                  'changed': float((initial != result.labels).mean()),
                  'pixel_energy_before': energy(problem, initial),
                  'pixel_energy_after': energy(problem, result.labels),
                  'hierarchy_energy_before': energies[0], 'hierarchy_energy_after': result.energy,
                  'trace': result.trace})
        print(f"DONE {record['index']} {result.seconds:.1f}s; hierarchy E "
              f'{energies[0]:.3f} -> {result.energy:.3f}', flush=True)


def completed_samples(directory, records):
    """Require the exact frozen population, rather than silently intersecting IDs."""
    expected = {r['index']: r['id'] for r in records}
    found = {}
    for path in directory.glob('*.json'):
        record = json.loads(path.read_text())
        index = record['index']
        if index not in expected or record['id'] != expected[index] or index in found:
            raise ValueError(f'Unexpected/duplicate sample result: {path}')
        if path.with_suffix('.npz').exists():
            found[index] = record
    return found, set(found) == set(expected)


def report(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    rows, scores = [], {}
    for treatment in protocol['treatments']:
        directory = root / 'runs' / treatment['name'] / 'dev' / 'samples'
        found, complete = completed_samples(directory, protocol['samples'])
        row = {'method': treatment['method'], 'mu': treatment['mu'],
               'samples': len(found), 'expected': len(protocol['samples']),
               'status': 'Complete' if complete else 'Incomplete'}
        if complete:
            baseline, refined = [], []
            for index in sorted(found):
                with np.load(directory / f'{index:04d}.npz') as z:
                    if str(z['id']) != found[index]['id']:
                        raise ValueError('Result array/metadata ID mismatch')
                    baseline.append(z['argmax_confusion'])
                    refined.append(z['wcsp_confusion'])
            baseline, refined = np.array(baseline), np.array(refined)
            base_score = pixel_report(baseline.sum(0))['mIoU']
            if not np.isclose(base_score, protocol['baseline']['mIoU'], atol=1e-12):
                raise ValueError('Argmax does not reproduce the frozen baseline')
            row.update(argmax_mIoU=base_score, wcsp_mIoU=pixel_report(refined.sum(0))['mIoU'],
                       delta_mIoU=pixel_report(refined.sum(0))['mIoU'] - base_score,
                       mean_seconds=float(np.mean([r['seconds'] for r in found.values()])))
            scores[(treatment['method'], treatment['mu'])] = refined
        rows.append(row)
    for row in rows:
        key = row['method'], row['mu']
        control = row['method'], 0
        if key in scores and control in scores:
            row['vs_mu0'] = paired_bootstrap(scores[control], scores[key],
                                            [r['roi'] for r in protocol['samples']])
    save_json(root / 'summary.json', rows)
    fields = ['method', 'mu', 'samples', 'expected', 'status', 'argmax_mIoU',
              'wcsp_mIoU', 'delta_mIoU', 'mean_seconds']
    with (root / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Full-resolution BCSS multiscale mu sweep', '',
             'Development selection only; no held-out validation is used to select mu.', '',
             '| Solver | mu | Finished / expected | mIoU | Mean s/tile | Status |',
             '|---|---:|---:|---:|---:|---|']
    for row in rows:
        metric = f"{row['wcsp_mIoU']:.6f}" if 'wcsp_mIoU' in row else '—'
        runtime = f"{row['mean_seconds']:.1f}" if 'mean_seconds' in row else '—'
        lines.append(f"| {row['method']} | {row['mu']} | {row['samples']} / {row['expected']} | "
                     f"{metric} | {runtime} | {row['status']} |")
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'solve', 'report'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--treatment', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=16)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.chdir(REPO)
    if args.command == 'prepare':
        prepare(args.root.resolve())
    elif args.command == 'solve':
        if not 0 <= args.shard < args.shards:
            parser.error('shard must be in 0..shards-1')
        solve(args.root.resolve(), args.treatment, args.shard, args.shards, args.smoke)
    else:
        report(args.root.resolve())


if __name__ == '__main__':
    main()
