"""Expanded full-512 development sweep with measured and audited convergence."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.bcss_wcsp.multiscale.convergence_solver import make_solver
from wcsp.config import ProblemConfig
from wcsp.evaluate.compare import paired_bootstrap
from wcsp.evaluate.pixel import pixel_confusion, pixel_report
from wcsp.model import build_problem
from wcsp.solve.energy import energy
from wcsp.solve.instrumentation import capture_events, emit, iteration_totals

SOURCE_ROOT = REPO / 'data/fullres_multiscale_20261115'
DEFAULT_ROOT = REPO / 'data/fullres_expanded_20261122_v2'
MUS, LAMBDAS = (0, .5, 1, 2, 4), (4, 8, 16)
BUDGETS = (4, 8, 12)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def treatments(base):
    rows = []
    # Preserve inner limits to obtain directly comparable 4/8/12 outer-round prefixes.
    for mu in (0, 1):
        cfg = copy.deepcopy(base)
        cfg['solver'].update(mu=mu, max_rounds=None, block_max_sweeps=50)
        rows.append(dict(name=f'outer_curve_mu{mu}', mode='outer_curve', mu=mu, lam=8, config=cfg))
    # Remove every iteration cap in the fully converged grid, including coarse ICM.
    for lam in LAMBDAS:
        for mu in MUS:
            cfg = copy.deepcopy(base)
            cfg['problem']['lam'] = lam
            cfg['solver'].update(mu=mu, block_lam=lam, max_rounds=None, block_max_sweeps=None)
            cfg['solver']['pixel']['max_rounds'] = None
            cfg['solver']['pixel']['inner']['max_sweeps'] = None
            rows.append(dict(name=f'converged_lam{lam}_mu{mu:g}', mode='full', mu=mu, lam=lam, config=cfg))
    return rows


def prepare(root, source_root):
    if (root/'protocol.json').exists():
        verify(root)
        print('Existing frozen protocol preserved.', flush=True)
        return
    source = json.loads((source_root/'protocol.json').read_text())
    if source['split'] != 'dev' or len(source['samples']) != 185:
        raise ValueError('Expected the original frozen 185-tile development population')
    records = source['samples']
    if len({r['id'] for r in records}) != 185 or len({r['index'] for r in records}) != 185:
        raise ValueError('Duplicate sample identifiers')
    baseline = np.zeros((5, 5), np.int64)
    for record in records:
        path = source_root/'inputs'/f"{record['index']:04d}.npz"
        if digest(path) != record['sha256']:
            raise ValueError(f'Frozen input checksum mismatch: {path}')
        with np.load(path) as z:
            probs, target = z['probs'], z['target']
            if str(z['id']) != record['id'] or probs.shape != (5, 512, 512) or target.shape != (512, 512):
                raise ValueError('Wrong input ID or full-resolution field')
            if not np.isfinite(probs).all() or not np.isin(target, [0, 1, 2, 3, 4, 255]).all():
                raise ValueError('Invalid input values')
            baseline += pixel_confusion(probs.argmax(0), target)
    baseline = pixel_report(baseline)
    if not np.isclose(baseline['mIoU'], source['baseline']['mIoU'], atol=1e-12):
        raise ValueError('Cached inputs do not reproduce the frozen argmax baseline')
    base = next(t['config'] for t in source['treatments'] if t['method'] == 'icm' and t['mu'] == 1)
    base = copy.deepcopy(base)
    base['paths']['runs'] = str(root/'runs')
    definitions = treatments(base)
    paths = list((REPO/'wcsp').rglob('*.py'))
    paths += [Path(__file__), Path(__file__).with_name('convergence_solver.py'), REPO/base['problem']['table']]
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), split='dev',
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_root=str(source_root), source_protocol_sha256=digest(source_root/'protocol.json'),
                    model_sha256=source['files']['data/outputs/tonight/full512/best.pt'],
                    samples=records, baseline=baseline, budgets=list(BUDGETS),
                    treatments=definitions, files={str(p.resolve().relative_to(REPO)): digest(p) for p in paths},
                    convergence='No improving single-variable move above 1e-9 at any hierarchy level; not a global optimum.',
                    allocation_limit='SLURM can terminate a task; unfinished tiles remain incomplete, with live traces retained.')
    for row in definitions:
        directory = root/'runs'/row['name']/'dev'
        (directory/'samples').mkdir(parents=True, exist_ok=True)
        save_json(directory/'config.json', row['config'])
    save_json(root/'protocol.json', protocol)
    print(f"Frozen {len(records)} dev inputs and {len(definitions)} treatments; argmax {baseline['mIoU']:.6f}", flush=True)


def verify(root):
    p = json.loads((root/'protocol.json').read_text())
    for relative, checksum in p['files'].items():
        if digest(REPO/relative) != checksum:
            raise ValueError(f'Experiment source changed: {relative}; use a new output root')
    if digest(Path(p['source_root'])/'protocol.json') != p['source_protocol_sha256']:
        raise ValueError('Original data protocol changed')
    return p


def save_arrays(path, record, initial, labels, levels, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f'.{os.getpid()}.partial.npz')
    arrays = dict(id=record['id'], argmax=initial.astype(np.uint8), wcsp=labels.astype(np.uint8),
                  argmax_confusion=pixel_confusion(initial, target), wcsp_confusion=pixel_confusion(labels, target))
    arrays.update({f'level_{i+1}': z.astype(np.uint8) for i, z in enumerate(levels)})
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def solve(root, treatment_index, shard, shards, smoke=False, replay=False):
    p = verify(root)
    treatment = p['treatments'][treatment_index]
    base = root/('smoke' if smoke else 'runs')/treatment['name']/'dev'
    (base/'samples').mkdir(parents=True, exist_ok=True)
    spec = copy.deepcopy(treatment['config']['solver'])
    if replay:
        spec.update(max_rounds=4, block_max_sweeps=50)
        spec['pixel']['max_rounds'] = 4
        spec['pixel']['inner']['max_sweeps'] = 50
    solver = make_solver(spec)
    cfg = ProblemConfig.from_dict(treatment['config']['problem'])
    records = p['samples'][:1] if smoke else p['samples'][shard::shards]
    for record in records:
        stem = f"{record['index']:04d}"
        destination = base/'samples'/f'{stem}.json'
        if destination.exists() and destination.with_suffix('.npz').exists():
            existing = json.loads(destination.read_text())
            if not replay and (not existing['converged'] or existing['replay']):
                raise ValueError('Existing output is not a converged result')
            continue
        path = Path(p['source_root'])/'inputs'/f'{stem}.npz'
        if digest(path) != record['sha256']:
            raise ValueError('Frozen probability/ground-truth input changed')
        with np.load(path) as z:
            probs, target = z['probs'].astype(np.float32), z['target']
        problem = build_problem(probs, cfg)
        initial = problem.unary.argmin(-1)
        attempt = uuid.uuid4().hex
        written = set()

        def checkpoint(budget, actual, labels, levels, stages, seconds):
            directory = base/'budgets'/str(budget)
            save_arrays(directory/f'{stem}.npz', record, initial, labels, levels, target)
            totals = iteration_totals(dict(stages=stages))
            save_json(directory/f'{stem}.json', dict(id=record['id'], index=record['index'],
                      requested_rounds=budget, actual_rounds=actual, seconds=seconds, **totals))
            written.add(budget)

        def on_round(round_, labels, levels, trace, stages, seconds):
            if treatment['mode'] == 'outer_curve' and round_ in BUDGETS:
                checkpoint(round_, round_, labels, levels, stages, seconds)

        print(f"START {treatment['name']} {stem}", flush=True)
        log = base/'samples'/f'{stem}.iterations.jsonl'
        with log.open('a', buffering=1) as stream:
            def sink(event):
                stream.write(json.dumps(dict(timestamp=time.time(), attempt_id=attempt,
                                             id=record['id'], **event)) + '\n')
            with capture_events(sink):
                emit('sample_start', mode=treatment['mode'], solver=spec)
                result = solver.solve(problem, initial, on_round)
                if not replay and not result.stats['converged']:
                    raise ValueError('Uncertified result; refusing to mark a full-convergence tile complete')
                totals = iteration_totals(result.stats)
                if treatment['mode'] == 'outer_curve':
                    for budget in BUDGETS:
                        if budget not in written and result.stats['rounds'] <= budget:
                            checkpoint(budget, result.stats['rounds'], result.labels,
                                       result.levels, result.stats['stages'], result.seconds)
                save_arrays(destination.with_suffix('.npz'), record, initial, result.labels, result.levels, target)
                if replay:
                    old = Path(p['source_root'])/'runs'/f"full512_ms_icm_mu{treatment['mu']:g}"/'dev/samples'/f'{stem}.npz'
                    with np.load(old) as z:
                        if not np.array_equal(result.labels, z['wcsp']):
                            raise ValueError('Instrumented replay differs from historical pixel labels')
                        for level, labels in enumerate(result.levels, start=1):
                            if not np.array_equal(labels, z[f'level_{level}']):
                                raise ValueError('Instrumented replay differs from historical block labels')
                record_out = dict(id=record['id'], index=record['index'], roi=record['roi'],
                                  seconds=result.seconds, rounds=result.stats['rounds'],
                                  replay=replay, converged=result.stats['converged'], **totals,
                                  audit=result.stats['audit'], stop_reason=result.stats['stop_reason'],
                                  changed=float((initial != result.labels).mean()),
                                  pixel_energy_before=energy(problem, initial), pixel_energy_after=energy(problem, result.labels),
                                  hierarchy_energy_before=result.trace[0]['energy'], hierarchy_energy_after=result.energy,
                                  trace=result.trace, stats=result.stats)
                save_json(destination, record_out)
                emit('sample_end', **{key: record_out[key] for key in ('rounds', 'converged', 'stop_reason')}, **totals)
        print(f"DONE {stem}: {result.stats['rounds']} outer rounds, "
              f"{totals['icm_sweeps']} ICM sweeps, {result.seconds:.1f}s, "
              f"audit={result.stats['audit']['improving_variables']}", flush=True)


def population(directory, expected):
    names = {r['index']: r['id'] for r in expected}
    files = {p.name for p in directory.iterdir()} if directory.exists() else set()
    found = {}
    for name in files:
        if not name.endswith('.json') or name[:-5]+'.npz' not in files:
            continue
        r = json.loads((directory/name).read_text())
        if r['index'] not in names or names[r['index']] != r['id'] or r['index'] in found:
            raise ValueError('Wrong or duplicate sample population')
        found[r['index']] = r
    return found, set(found) == set(names)


def aggregate(directory, records, require_audit, baseline):
    found, complete = population(directory, records)
    row = dict(samples=len(found), expected=len(records), status='Complete' if complete else 'Incomplete')
    if not complete:
        return row, None
    if require_audit and any(not r['converged'] or r['audit']['improving_variables'] for r in found.values()):
        raise ValueError('Complete convergence report contains an unaudited result')
    before, after = [], []
    for index in sorted(found):
        with np.load(directory/f'{index:04d}.npz') as z:
            if str(z['id']) != found[index]['id']:
                raise ValueError('Result array ID mismatch')
            before.append(z['argmax_confusion']); after.append(z['wcsp_confusion'])
    before, after = np.array(before), np.array(after)
    if not np.isclose(pixel_report(before.sum(0))['mIoU'], baseline, atol=1e-12):
        raise ValueError('Input baseline mismatch')
    row.update(metrics=pixel_report(after.sum(0)),
               mean_seconds=float(np.mean([r['seconds'] for r in found.values()])),
               mean_rounds=float(np.mean([r.get('rounds', r.get('actual_rounds')) for r in found.values()])),
               total_icm_sweeps=sum(r['icm_sweeps'] for r in found.values()),
               total_icm_calls=sum(r['icm_calls'] for r in found.values()),
               icm_cap_hits=sum(r['icm_sweep_limit_hits'] for r in found.values()))
    return row, after


def report(root):
    p = json.loads((root/'protocol.json').read_text())
    rows, matrices = [], {}
    for treatment in p['treatments']:
        base = root/'runs'/treatment['name']/'dev'
        entries = [('converged', base/'samples', True)]
        if treatment['mode'] == 'outer_curve':
            entries += [(f'rounds_{budget}', base/'budgets'/str(budget), False) for budget in BUDGETS]
        for label, directory, audit in entries:
            row, mat = aggregate(directory, p['samples'], audit, p['baseline']['mIoU'])
            row.update(name=treatment['name'], mode=treatment['mode'], mu=treatment['mu'],
                       lam=treatment['lam'], checkpoint=label)
            rows.append(row)
            if mat is not None:
                matrices[(treatment['name'], label)] = mat
    groups = [r['roi'] for r in p['samples']]
    for row in rows:
        key = row['name'], row['checkpoint']
        if key not in matrices:
            continue
        control = (f"converged_lam{row['lam']}_mu0", 'converged') if row['mode'] == 'full' else ('outer_curve_mu0', row['checkpoint'])
        if control in matrices:
            row['vs_mu0'] = paired_bootstrap(matrices[control], matrices[key], groups)
        if row['mode'] == 'outer_curve' and row['checkpoint'] == 'rounds_4':
            previous = json.loads((Path(p['source_root'])/'summary.json').read_text())
            previous = next(r for r in previous if r['method'] == 'icm' and r['mu'] == row['mu'])
            if not np.isclose(row['metrics']['mIoU'], previous['wcsp_mIoU'], atol=1e-12):
                raise ValueError('Four-round replay failed historical aggregate parity')
    save_json(root/'summary.json', dict(split='dev', baseline=p['baseline'], results=rows))
    lines = ['# Expanded full-resolution BCSS dev sweep', '',
             'Fully converged results require a single-variable stationarity audit at every level. '
             'No new validation evaluation or model inference is performed.', '',
             '| Run | Checkpoint | Finished | mIoU | Mean outer rounds | Mean s/tile | ICM sweeps | Status |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        metric = f"{row['metrics']['mIoU']:.6f}" if 'metrics' in row else '—'
        rounds = f"{row['mean_rounds']:.1f}" if 'mean_rounds' in row else '—'
        seconds = f"{row['mean_seconds']:.1f}" if 'mean_seconds' in row else '—'
        lines.append(f"| {row['name']} | {row['checkpoint']} | {row['samples']}/{row['expected']} | "
                     f"{metric} | {rounds} | {seconds} | {row.get('total_icm_sweeps', '—')} | {row['status']} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'solve', 'report'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--treatment', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=32)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or (args.replay and not args.smoke):
        parser.error('Invalid shard, or replay without --smoke')
    os.chdir(REPO)
    if args.command == 'prepare':
        prepare(args.root.resolve(), args.source_root.resolve())
    elif args.command == 'solve':
        solve(args.root.resolve(), args.treatment, args.shard, args.shards, args.smoke, args.replay)
    else:
        report(args.root.resolve())


if __name__ == '__main__':
    main()
