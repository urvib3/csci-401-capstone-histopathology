"""Freeze and run BCSS temperature, context, region-move and appearance experiments."""
import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.bcss_wcsp.multiscale.fullres_expanded import digest, save_json, save_arrays, population
from experiments.bcss_wcsp.multiscale.convergence_solver import make_solver, MeasuredResult
from experiments.bcss_wcsp.multiscale.correction_methods import (
    appearance_features, appearance_logits, region_appearance_unary,
    presence_problem, solve_with_regions, edit_counts)
from wcsp.config import ProblemConfig
from wcsp.data.bcss import BCSS
from wcsp.evaluate.compare import paired_bootstrap
from wcsp.evaluate.pixel import pixel_confusion, pixel_report
from wcsp.labels import BCSS_TO_WCSP, remap_labels
from wcsp.model import build_problem
from wcsp.solve.energy import energy
from wcsp.solve.instrumentation import capture_events, emit, iteration_totals

SOURCE = REPO/'data/fullres_multiscale_20261115'
ROOT = REPO/'data/fullres_corrections_20261122'
DATA = REPO/'data/bcss'


def treatments(base):
    rows = []
    def add(name, family, mu, **options):
        cfg = copy.deepcopy(base)
        cfg['problem'].update(lam=8, temperature=options.pop('temperature', 1))
        cfg['solver'].update(mu=mu, block_lam=8, max_rounds=None, block_max_sweeps=None,
                             block_alpha=options.pop('block_alpha', 1))
        cfg['solver']['pixel']['max_rounds'] = None
        cfg['solver']['pixel']['inner']['max_sweeps'] = None
        rows.append(dict(name=name, family=family, mu=mu, config=cfg,
                         control=f'temp1_mu{mu}', **options))
    for t in (1, 1.5, 2, 3, 4):
        for mu in (0, 1):
            add(f'temp{t:g}_mu{mu}', 'temperature', mu, temperature=t)
    for alpha in (0, .25, .5):
        add(f'blockalpha{alpha:g}_mu1', 'block_fidelity', 1, block_alpha=alpha)
    for gamma in (40, 120, 400, 1200, 4000, 12000):
        for mu in (0, 1):
            add(f'presence{gamma}_mu{mu}', 'presence', mu, gamma=gamma)
    for method in ('components', 'slic'):
        for mu in (0, 1):
            add(f'regions_{method}_mu{mu}', 'region_moves', mu, regions=method)
    for beta in (1, 4):
        for mu in (0, 1):
            add(f'hedcontrast{beta}_mu{mu}', 'image_boundary', mu)
            rows[-1]['config']['problem'].update(contrast_beta=beta, color_space='hed')
    for weight in (.1, .3, 1):
        for mu in (0, 1):
            add(f'appearance{weight:g}_mu{mu}', 'regional_evidence', mu, appearance_weight=weight)
    add('appearance_only', 'appearance_control', 0)
    return rows


def source_files():
    paths = list((REPO/'wcsp').rglob('*.py'))
    paths += list((REPO/'benchmarks/bcss/loader').glob('*.py'))
    paths += [Path(__file__), Path(__file__).with_name('correction_methods.py'),
              Path(__file__).with_name('convergence_solver.py'),
              Path(__file__).with_name('fullres_expanded.py'),
              REPO/'experiments/bcss_wcsp/configs/tables/bcss_elicited.csv']
    return {str(p.relative_to(REPO)): digest(p) for p in paths}


def checked_mask(info):
    from PIL import Image
    from benchmarks.bcss.loader import coarse_lut
    data = Path(info.mask_path).read_bytes()
    mask = np.array(Image.open(io.BytesIO(data)))
    if mask.ndim != 2:
        raise ValueError('Expected a grayscale BCSS annotation')
    target = remap_labels(coarse_lut()[0][mask.astype(np.uint8)], BCSS_TO_WCSP)
    return target, digest(info.mask_path)


def fit_assets(data_root):
    """Full-field presence prior and an independent, fit-only appearance model."""
    from concurrent.futures import ThreadPoolExecutor
    fit = BCSS('fit', root=str(data_root), full_tile=True)
    dev = BCSS('dev', root=str(data_root), full_tile=True)
    if set(fit.ids) & set(dev.ids):
        raise ValueError('Overlapping fitting/development IDs')
    fit_rois = {fit.data.tiles[i].roi_id for i in fit.positions}
    dev_rois = {dev.data.tiles[i].roi_id for i in dev.positions}
    if fit_rois & dev_rois:
        raise ValueError('Overlapping fitting/development ROIs')
    counts = np.zeros(32, np.int64)
    records = []
    infos = [fit.data.tiles[i] for i in fit.positions]
    with ThreadPoolExecutor(max_workers=4) as pool_:
        for info, (target, checksum) in zip(infos, pool_.map(checked_mask, infos)):
            valid = target != 255
            if valid.sum() >= 1000:
                share = np.bincount(target[valid], minlength=5) / valid.sum()
                mask = sum(1 << k for k in range(5) if share[k] >= .01)
                counts[mask] += 1
            records.append(dict(id=info.name.removesuffix('.png'), roi=info.roi_id,
                                mask_sha256=checksum))
    print(f'Presence prior fitted on {int(counts.sum())}/{len(fit)} full-field fit tiles', flush=True)
    prior = np.zeros(32)
    prior[1:] = -np.log((counts[1:] + 1) / (counts[1:] + 1).sum())
    # 288 evenly spaced fit tiles: every ninth is reserved for fit-only calibration.
    positions = np.linspace(0, len(fit)-1, min(288, len(fit))).round().astype(int)
    number, sums, squares = np.zeros(5), np.zeros((5, 9)), np.zeros((5, 9))
    calibration, feature_manifest = [], []
    for order, position in enumerate(positions):
        sample = fit[int(position)]
        features = appearance_features(sample.image)
        rng = np.random.default_rng(20261122 + int(position))
        purpose = 'calibration' if order % 9 == 0 else 'fit'
        info = fit.data.tiles[fit.positions[int(position)]]
        feature_manifest.append(dict(id=sample.id, purpose=purpose,
                                     image_sha256=digest(info.image_path), mask_sha256=digest(info.mask_path)))
        if purpose == 'calibration':
            valid = np.flatnonzero(sample.target.ravel() != 255)
            ids = rng.choice(valid, min(2000, len(valid)), replace=False)
            if len(ids):
                calibration.append((features.reshape(-1, 9)[ids], sample.target.ravel()[ids]))
        else:
            for k in range(5):
                ids = np.flatnonzero(sample.target.ravel() == k)
                ids = rng.choice(ids, min(128, len(ids)), replace=False)
                x = features.reshape(-1, 9)[ids].astype(np.float64)
                number[k] += len(x); sums[k] += x.sum(0); squares[k] += (x*x).sum(0)
        if order % 32 == 0:
            print(f'Appearance fit/calibration tile {order+1}/{len(positions)}', flush=True)
    if (number == 0).any() or not calibration:
        raise ValueError('Insufficient fitting data for all five appearance classes')
    means = sums / number[:, None]
    variances = np.maximum(squares / number[:, None] - means ** 2, 1e-6)
    variances = np.maximum(variances, .05 * variances.mean(0))
    model = dict(means=means.tolist(), variances=variances.tolist(),
                 sampled_class_counts=number.astype(int).tolist(), class_priors='uniform',
                 features='HED + Gaussian sigma4 mean + local standard deviation',
                 temperature=1)
    scores = np.concatenate([appearance_logits(x, model) for x, _ in calibration])
    truth = np.concatenate([y for _, y in calibration])
    from scipy.special import logsumexp
    nll = {}
    for t in (.5, 1, 2, 4, 8, 16, 32):
        scaled = scores / t
        nll[t] = float(np.mean(logsumexp(scaled, axis=-1) - scaled[np.arange(len(truth)), truth]))
    model.update(temperature=min(nll, key=nll.get), fit_calibration_nll=nll,
                 fit_calibration_pixels=len(truth), manifest=feature_manifest)
    return prior.tolist(), counts.tolist(), records, model, dev


def prepare(root, data_root):
    if (root/'protocol.json').exists():
        verify(root)
        print('Existing correction protocol preserved.', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    snapshot = source_files()
    source = json.loads((SOURCE/'protocol.json').read_text())
    if len(source['samples']) != 185 or source['split'] != 'dev':
        raise ValueError('Expected the frozen 185-tile development population')
    prior, counts, fit_records, appearance, dev = fit_assets(data_root)
    save_json(root/'fit_manifest.json', fit_records)
    save_json(root/'appearance_model.json', appearance)
    save_json(root/'presence_prior.json', dict(cost=prior, counts=counts,
              threshold=.01, min_labeled=1000, field='full512', split='fit'))
    asset_records = []
    from skimage.segmentation import slic
    for record in source['samples']:
        source_path = SOURCE/'inputs'/f"{record['index']:04d}.npz"
        if digest(source_path) != record['sha256']:
            raise ValueError('Frozen input changed')
        sample = dev[dev.index(record['id'])]
        with np.load(source_path) as z:
            if not np.array_equal(sample.target, z['target']):
                raise ValueError(f'Current RGB/annotation does not align with frozen inputs: {sample.id}')
        segments = slic(sample.image, n_segments=256, compactness=10, sigma=1,
                        start_label=1, channel_axis=-1, enforce_connectivity=True)
        unary = region_appearance_unary(sample.image, segments, appearance)
        directory = root/'assets'
        directory.mkdir(exist_ok=True)
        path = directory/f"{record['index']:04d}.npz"
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, id=sample.id, image=sample.image,
                            segments=segments.astype(np.int32), appearance_unary=unary)
        temporary.replace(path)
        info = dev.data.tiles[dev.positions[dev.index(sample.id)]]
        asset_records.append(dict(index=record['index'], id=sample.id, sha256=digest(path),
                                  image_sha256=digest(info.image_path)))
        if record['index'] % 16 == 0:
            print(f"Cached aligned RGB/SLIC/appearance {record['index']+1}/185", flush=True)
    if source_files() != snapshot:
        raise ValueError('Source changed during preparation; use a new root')
    base = next(t['config'] for t in source['treatments'] if t['method']=='icm' and t['mu']==1)
    definitions = treatments(base)
    for t in definitions:
        save_json(root/'runs'/t['name']/'dev/config.json', t)
    files = snapshot | {str((root/name).relative_to(REPO)): digest(root/name) for name in
                        ('appearance_model.json', 'presence_prior.json', 'fit_manifest.json')}
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), split='dev',
                    source_root=str(SOURCE), source_protocol_sha256=digest(SOURCE/'protocol.json'),
                    model_sha256=source['files']['data/outputs/tonight/full512/best.pt'],
                    samples=source['samples'], baseline=source['baseline'], files=files,
                    assets=asset_records, treatments=definitions,
                    class_order=['tumor','inflammatory','stroma','necrosis','other'],
                    fit_root=str(data_root), fitting_tiles=len(fit_records),
                    region_budget=dict(passes=2, maximum_regions=32, minimum_pixels=64),
                    selection='Development only; no validation inputs used.',
                    temperature_semantics='Temperature and appearance affect coarse evidence reconstructed from pixel unaries.')
    save_json(root/'protocol.json', protocol)
    print(f'Frozen {len(definitions)} correction treatments on 185 development tiles.', flush=True)


def verify(root):
    p = json.loads((root/'protocol.json').read_text())
    for name, checksum in p['files'].items():
        if digest(REPO/name) != checksum:
            raise ValueError(f'Frozen correction source/fit artifact changed: {name}')
    if digest(Path(p['source_root'])/'protocol.json') != p['source_protocol_sha256']:
        raise ValueError('Frozen input protocol changed')
    return p


def solve(root, treatment_index, shard=0, shards=32, smoke=False):
    p = verify(root)
    t = p['treatments'][treatment_index]
    base = root/('smoke' if smoke else 'runs')/t['name']/'dev/samples'
    base.mkdir(parents=True, exist_ok=True)
    prior = json.loads((root/'presence_prior.json').read_text())['cost']
    assets = {a['index']: a for a in p['assets']}
    records = p['samples'][:1] if smoke else p['samples'][shard::shards]
    for r in records:
        path = base/f"{r['index']:04d}.json"
        if path.exists() and path.with_suffix('.npz').exists():
            old = json.loads(path.read_text())
            if not old['converged'] or old['id'] != r['id']:
                raise ValueError('Invalid existing completion record')
            continue
        source = Path(p['source_root'])/'inputs'/f"{r['index']:04d}.npz"
        if digest(source) != r['sha256']:
            raise ValueError('Frozen input checksum mismatch')
        with np.load(source) as z:
            probs, target = z['probs'].astype(np.float32), z['target']
        asset = root/'assets'/f"{r['index']:04d}.npz"
        if digest(asset) != assets[r['index']]['sha256']:
            raise ValueError('RGB/region asset checksum mismatch')
        with np.load(asset) as z:
            if str(z['id']) != r['id']:
                raise ValueError('Image asset ID mismatch')
            image, segments, appearance = z['image'], z['segments'], z['appearance_unary']
        initial = probs.argmax(0)
        problem = build_problem(probs, ProblemConfig.from_dict(t['config']['problem']), image=image)
        if t.get('appearance_weight'):
            problem = replace(problem, unary=problem.unary + t['appearance_weight'] * appearance)
        presence = None
        if t['family']=='presence':
            problem, presence = presence_problem(problem, prior, t['gamma'])
        init = problem.unary.argmin(-1)
        solver = make_solver(t['config']['solver'])
        attempt = uuid.uuid4().hex
        print(f"START {t['name']} {r['index']:04d}", flush=True)
        with path.with_suffix('.iterations.jsonl').open('a', buffering=1) as stream:
            def sink(e):
                stream.write(json.dumps(dict(timestamp=time.time(), attempt_id=attempt, id=r['id'], **e))+'\n')
            with capture_events(sink):
                emit('sample_start', treatment=t['name'])
                if t['family']=='appearance_control':
                    before = time.perf_counter()
                    labels = appearance.argmin(-1)
                    problem = replace(problem, unary=appearance, weights=np.zeros_like(problem.weights))
                    result = MeasuredResult(labels, energy(problem, labels), time.perf_counter()-before,
                        levels=[], stats=dict(converged=True, audit=dict(improving_variables=0),
                        rounds=0, stages=[], stop_reason='independent_regional_unary_optimum'))
                elif t['family']=='region_moves':
                    result = solve_with_regions(solver, problem, init,
                                                segments if t['regions']=='slic' else None)
                else:
                    result = solver.solve(problem, init)
                if not result.stats['converged'] or result.stats['audit']['improving_variables']:
                    raise ValueError('Refusing an unaudited completion')
                if presence is not None:
                    presence['forbidden_pixels'] = int(sum((result.labels==k).sum()
                        for k in range(5) if not presence['chosen'] & (1<<k)))
                    if presence['forbidden_pixels']:
                        raise ValueError('Presence exclusion penalty was insufficient')
                save_arrays(path.with_suffix('.npz'), r, initial, result.labels, result.levels, target)
                out = dict(id=r['id'], index=r['index'], roi=r['roi'], treatment=t['name'],
                           converged=True, seconds=result.seconds, rounds=result.stats['rounds'],
                           stop_reason=result.stats['stop_reason'], audit=result.stats['audit'],
                           edits=edit_counts(initial, result.labels, target), presence=presence,
                           pixel_energy_before=energy(problem, init), pixel_energy_after=energy(problem, result.labels),
                           hierarchy_energy_after=result.energy, **iteration_totals(result.stats),
                           stats=result.stats)
                save_json(path, out)
                emit('sample_end', seconds=result.seconds, converged=True)
        print(f"DONE {r['index']:04d}: {result.seconds:.1f}s, fixes={out['edits']['all']['fixes']}, breaks={out['edits']['all']['breaks']}", flush=True)


def report(root):
    p = verify(root)
    rows, matrices = [], {}
    for t in p['treatments']:
        directory = root/'runs'/t['name']/'dev/samples'
        found, complete = population(directory, p['samples'])
        row = dict(name=t['name'], family=t['family'], samples=len(found), expected=len(p['samples']),
                   status='Complete' if complete else 'Incomplete', control=t['control'])
        if complete:
            if any(not r['converged'] or r['audit']['improving_variables'] for r in found.values()):
                raise ValueError('Unaudited complete population')
            before, after = [], []
            for index in sorted(found):
                with np.load(directory/f'{index:04d}.npz') as z:
                    if str(z['id']) != found[index]['id']:
                        raise ValueError('Report input ID mismatch')
                    before.append(z['argmax_confusion']); after.append(z['wcsp_confusion'])
            before, after = np.array(before), np.array(after)
            if not np.isclose(pixel_report(before.sum(0))['mIoU'], p['baseline']['mIoU'], atol=1e-12):
                raise ValueError('Baseline population mismatch')
            edits = {group: {key:sum(r['edits'][group][key] for r in found.values())
                            for key in ('fixes','breaks','changed','wrong_to_wrong')}
                     for group in ('all','boundary_10px','interior_over_10px')}
            row.update(metrics=pixel_report(after.sum(0)), edits=edits,
                       mean_seconds=float(np.mean([r['seconds'] for r in found.values()])),
                       vs_unet=paired_bootstrap(before, after, [r['roi'] for r in p['samples']]))
            matrices[t['name']] = after
        rows.append(row)
    for row in rows:
        if row['name'] in matrices and row['control'] in matrices:
            row['vs_control'] = paired_bootstrap(matrices[row['control']], matrices[row['name']],
                                                 [r['roi'] for r in p['samples']])
    save_json(root/'summary.json', dict(split='dev', baseline=p['baseline'], results=rows))
    lines=['# BCSS full-512 correction experiments', '',
           'Development only. Partial populations receive no aggregate accuracy. Region proposals are budgeted; stationarity audits cover single variables.', '',
           '| Treatment | Finished | mIoU | Fixes | Breaks | Mean s/tile | Status |',
           '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        metric=f"{r['metrics']['mIoU']:.6f}" if 'metrics' in r else '—'
        edits=r.get('edits',{}).get('all',{})
        seconds=f"{r['mean_seconds']:.1f}" if 'mean_seconds' in r else '—'
        lines.append(f"| {r['name']} | {r['samples']}/{r['expected']} | {metric} | {edits.get('fixes','—')} | {edits.get('breaks','—')} | {seconds} | {r['status']} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','solve','report','verify','smoke'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--treatment', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=32)
    parser.add_argument('--smoke', action='store_true')
    args=parser.parse_args()
    os.chdir(REPO)
    root=args.root.resolve()
    if not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    if args.command=='prepare': prepare(root, args.data_root.resolve())
    elif args.command=='solve': solve(root,args.treatment,args.shard,args.shards,args.smoke)
    elif args.command=='report': report(root)
    elif args.command=='verify':
        p=verify(root)
        print(f"Verified {len(p['treatments'])} treatments and source/fit checksums")
    else:
        p=verify(root)
        # Exercise changed objective, region moves and independent image evidence
        # on a full-512 tile before releasing the dependent production array.
        names=('temp2_mu1','presence4000_mu1','regions_slic_mu1','appearance0.3_mu1')
        for name in names:
            solve(root,next(i for i,t in enumerate(p['treatments']) if t['name']==name),smoke=True)
        save_json(root/'smoke_passed.json',dict(treatments=list(names),passed=True))


if __name__=='__main__':
    main()
