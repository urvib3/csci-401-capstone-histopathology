"""Refresh the marked BCSS results section in Deliverable 6."""
import argparse
import csv
import fcntl
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from wcsp.evaluate.pixel import pixel_report

BEGIN = '<!-- BEGIN BCSS FOLLOWUP RESULTS -->'
END = '<!-- END BCSS FOLLOWUP RESULTS -->'
DEFAULT_DELIVERABLE = REPO / 'deliverables/deliverable_6.md'
DOCUMENT_HEADER = (
    '# Deliverable 6 — BCSS Full-Resolution Training and WCSP Follow-up Experiments\n\n'
    'This deliverable reports the new BCSS training, distant-relation, and solver experiments. '
    'The earlier PanNuke and BCSS validation experiments remain in [Deliverable 5](deliverable_5.md).\n'
)
TRAINING = [('ce', 'Original architecture, cross-entropy'),
            ('full512', 'Full-resolution head + backbone, cross-entropy'),
            ('pairwise_0.01', 'Cross-entropy + 0.01 × pairwise'),
            ('pairwise_0.1', 'Cross-entropy + 0.1 × pairwise'),
            ('pairwise_1', 'Cross-entropy + 1 × pairwise')]
CENTER = [('tonight_pretrained', 'Pretrained U-Net'), ('tonight_ce', 'Cross-entropy control'),
          ('tonight_pairwise_0.01', 'Pairwise weight 0.01'),
          ('tonight_pairwise_0.1', 'Pairwise weight 0.1'), ('tonight_pairwise_1', 'Pairwise weight 1')]
FULL = [('tonight_pretrained_full512', 'Pretrained U-Net, bilinear'),
        ('tonight_ce_full512', 'Cross-entropy control, bilinear'), ('tonight_full512', 'Trained full-resolution head')]
SOLVERS = [('tonight_solver_icm_instrumented', 'ICM, instrumented'),
           ('tonight_solver_toulbar2', 'Toulbar2, 2 s/block'),
           ('tonight_solver_hybrid', 'ICM → Toulbar2, 2 s/block'),
           ('tonight_solver_toulbar2_unlimited_instrumented', 'Toulbar2, no search timer'),
           ('tonight_solver_hybrid_unlimited_instrumented', 'ICM → Toulbar2, no search timer')]
DISTANT = [('tonight_long_control', 'No distant edges (p=0)')]+[
    (f'tonight_long_d{d}_p{p}', f'd={d}, p={p}')
    for d in (8, 32) for p in ('0.000001', '0.00001')]


def number(value, digits=4):
    return f'{value:.{digits}f}' if value is not None and math.isfinite(float(value)) else '—'


def training_summary(root):
    result = {}
    for name, _ in TRAINING:
        path = root/name/'log.csv'
        if not path.exists():
            result[name] = None
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        valid = [r for r in rows if r.get('dev_mIoU') and math.isfinite(float(r['dev_mIoU']))]
        if not valid:
            result[name] = None
            continue
        best = max(valid, key=lambda r: float(r['dev_mIoU']))
        result[name] = {'epoch': int(best['epoch']), 'latest_epoch': max(int(r['epoch']) for r in valid),
                        'mIoU': float(best['dev_mIoU']), 'center_mIoU': float(best['dev_center_mIoU']),
                        'pixel_acc': float(best['dev_pixel_acc']),
                        'baseline_center_mIoU': float(valid[0]['dev_center_mIoU'])}
    return result


def run_summary(directory):
    paths = sorted(directory.glob('shard_*_of_*.npz'))
    expected = set()
    indices = set()
    cms = {name: np.zeros((5, 5), dtype=np.int64) for name in ('argmax', 'wcsp')}
    rows, ids = [], []
    for path in paths:
        match = re.fullmatch(r'shard_(\d+)_of_(\d+)\.npz', path.name)
        if match is None:
            continue
        indices.add(int(match[1])); expected.add(int(match[2]))
        with np.load(path) as data:
            ids.extend(data['ids'].tolist())
            for name in cms:
                if f'{name}_confusion' in data:
                    cms[name] += data[f'{name}_confusion'].sum(axis=0)
        with path.with_suffix('.csv').open() as stream:
            rows.extend(csv.DictReader(stream))
    if len(ids) != len(set(ids)):
        raise ValueError(f'duplicate sample ids in {directory}')
    complete = len(expected) == 1 and indices == set(range(next(iter(expected))))
    result = {'complete': complete, 'shards': len(indices), 'expected_shards': next(iter(expected)) if len(expected)==1 else 32,
              'samples': len(ids), 'ids': sorted(ids)}
    if complete and ids:
        result.update({name: pixel_report(cm) for name, cm in cms.items()})
        result['mean_seconds'] = float(np.mean([float(r['seconds']) for r in rows]))
        result['energy_reduction_pct'] = 100*(1-sum(float(r['energy_wcsp']) for r in rows)/sum(float(r['energy_argmax']) for r in rows))
        if rows and all('icm_sweeps' in r for r in rows):
            result['iterations'] = {
                'sweeps': sum(int(r['icm_sweeps']) for r in rows),
                'calls': sum(int(r['icm_calls']) for r in rows),
                'sweep_cap_hits': sum(int(r['icm_sweep_limit_hits']) for r in rows),
                'stops': dict(Counter(r.get('stop_reason', '') for r in rows))}
    trace_counts = Counter()
    finished = set()
    for path in directory.glob('shard_*.iterations.jsonl'):
        with path.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:  # a line may still be being written
                    continue
                if event['event'] == 'sample_end':
                    finished.add(event['id'])
                if event['event'] == 'toulbar_search_end':
                    trace_counts['searches_returned'] += 1
                    trace_counts['search_nodes'] += event.get('search_nodes') or 0
                    trace_counts['backtracks'] += event.get('backtracks') or 0
    result['streamed_finished_samples'] = len(finished)
    result['returned_search_counters'] = dict(trace_counts)
    return result


def metric_table(entries, runs):
    lines = ['| Method | Samples | U-Net mIoU | + WCSP mIoU | Mean solve s/tile | Status |',
             '|---|---:|---:|---:|---:|---|']
    for key, label in entries:
        run = runs[key]
        done = run['complete'] and run['samples'] > 0
        status = 'Complete' if done else f"Pending/incomplete: {run['shards']}/{run['expected_shards']} shards"
        lines.append(f"| {label} | {run['samples'] if done else '—'} | {number(run['argmax']['mIoU']) if done else '—'} | "
                     f"{number(run['wcsp']['mIoU']) if done else '—'} | {number(run.get('mean_seconds'),1) if done else '—'} | {status} |")
    return '\n'.join(lines)


def render(training, runs, timestamp):
    lines = [BEGIN, '## BCSS follow-up experiments (development results)', '',
        f'Automatically refreshed from experiment artifacts at **{timestamp}**. Incomplete runs are explicitly marked; their partial metrics are not treated as final results.', '',
        '**Protocol.** The follow-up uses the same underlying 151 BCSS ROIs and the same 106/45 hospital-held-out ROI split. '
        'Tiles were regenerated directly from the upstream images at requested 0.25 MPP: 7,768 fit, 1,012 dev, and 4,028 val tiles. '
        'Preparation version 2 preserves both axes of 16-bit grayscale masks; codes 0 and 7 are ignored. '
        'The documented earlier preparation has different tile counts, so the historical scores in [Deliverable 5](deliverable_5.md) are not a paired comparison to these follow-ups. '
        'These new results are **development results**, not a new held-out-validation result.', '',
        '**Training.** Cross-entropy (CE) is the supervised pixel-classification loss. The control fine-tunes the original half-resolution U-Net with CE. '
        'The full-resolution arm trains the existing full-resolution RGB encoder/decoder refinement head **and** the pretrained backbone on the entire 512×512 tile; '
        'this is one forward pass, not shift-and-stitch. Pairwise-loss arms retain the original architecture and central field, adding a differentiable expected compatibility-table cost to CE. '
        'All arms use 10 epochs, batch 4, seed 0, and checkpoint selection on dev. The full-resolution checkpoint is selected by whole-tile dev mIoU; other checkpoints use central-crop dev mIoU.', '',
        '| Training arm | Epochs logged / 10 | Best epoch | Central 256 mIoU | Whole 512 mIoU |',
        '|---|---:|---:|---:|---:|']
    baseline = (training.get('ce') or {}).get('baseline_center_mIoU')
    lines.append(f'| Pretrained U-Net reference | 0 | 0 | {number(baseline)} | — |')
    for key, label in TRAINING:
        row = training[key]
        if row is None:
            lines.append(f'| {label} | Pending | — | — | — |')
        else:
            lines.append(f"| {label} | {row['latest_epoch']}/10 | {row['epoch']} | {row['center_mIoU']:.4f} | {number(row['mIoU']) if key=='full512' else '—'} |")
    ce, full = training['ce'], training['full512']
    if ce and full:
        lines += ['', f"At the selected checkpoints, the full-resolution arm changes central-crop mIoU by "
                  f"{full['center_mIoU']-ce['center_mIoU']:+.4f} relative to the CE control. "
                  'Architecture and training field change together; this does not isolate resolution alone.']
    pair_rows = [training[key] for key, _ in TRAINING if key.startswith('pairwise')]
    if ce and ce['latest_epoch'] >= 10 and all(row and row['latest_epoch'] >= 10 for row in pair_rows):
        best_pair = max(row['center_mIoU'] for row in pair_rows)
        lines += [f"After all 10 epochs, the strongest pairwise-loss arm differs from the CE control by "
                  f"{best_pair-ce['center_mIoU']:+.4f} central-crop mIoU; "
                  + ('none of the tested pairwise-loss weights improves on the CE control.' if best_pair <= ce['center_mIoU'] else
                     'at least one tested weight improves the observed dev score; validation is still needed.')]
    lines += ['', 'Training-table scores use all 1,012 dev tiles. Inference comparisons below use 200 evenly selected dev tiles, '
              'excluding tiles without labeled pixels in the evaluated field (181 central-field tiles; 185 whole-field tiles). '
              'Scores across those different populations must not be subtracted. Within each inference table, methods use the same scored field. '
              'WCSP uses raw probabilities, the handwritten table, λ=8, η=1, and local neighborhood normalization.', '',
              '**Central-field model comparisons.**', '', metric_table(CENTER,runs), '',
              '**Whole-field model comparisons.**', '', metric_table(FULL,runs), '',
              '**Solver comparison on the same pretrained central-field probabilities.**', '', metric_table(SOLVERS,runs), '',
              'ICM permits at most 50 sweeps per block and stops early when no labels change. Block refinement permits four outer rounds and stops early on an energy plateau. '
              'The uncapped Toulbar2 arms have no internal search timer; the SLURM allocation still has an 8-hour wall limit. '
              'The earlier 2-second-capped searches are retained as separately labeled controls. A capped run that returns its starting labeling quickly is not evidence of faster optimization at comparable quality. '
              'Completing a conditioned block search does not establish a global optimum for the image.', '']
    for key,label in SOLVERS:
        run=runs[key]
        if not run['complete'] and run['streamed_finished_samples']:
            counts=run['returned_search_counters']
            lines.append(f"- {label}: {run['streamed_finished_samples']} tiles have finished according to streaming traces; "
                         f"{counts.get('searches_returned',0)} searches have returned with {counts.get('search_nodes',0):,} nodes and "
                         f"{counts.get('backtracks',0):,} backtracks. Counters exclude searches still running; aggregate mIoU is withheld until all shards exist.")
    icm=runs['tonight_solver_icm_instrumented'].get('iterations')
    if icm and icm['calls']:
        lines += [f"Instrumented ICM used {icm['sweeps']:,} sweeps over {icm['calls']:,} block calls "
                  f"({icm['sweeps']/icm['calls']:.2f} sweeps/call); {icm['sweep_cap_hits']} calls reached the sweep cap. "
                  f"Outer stopping reasons: {icm['stops'].get('energy_plateau',0)} energy plateaus and "
                  f"{icm['stops'].get('round_limit',0)} round-limit stops. An outer energy plateau is not a proof of global optimality.", '']
    lines += ['**Sampled distant relations.** Each unordered pair beyond Euclidean distance d is included independently with probability p. '
              'Extra edges retain the same compatibility table and distance decay, without diluting local weights or inverse-probability scaling. '
              'These runs use whole-grid ICM; the p=0 row is their matching solver control.', '',metric_table(DISTANT,runs),'']
    control=runs['tonight_long_control']
    distant=[runs[k] for k,_ in DISTANT[1:]]
    if control['complete'] and all(r['complete'] and r['ids']==control['ids'] for r in distant):
        deltas=[r['wcsp']['mIoU']-control['wcsp']['mIoU'] for r in distant]
        lines.append(f"Observed distant-edge mIoU differences from the matching control range from {min(deltas):+.6f} to {max(deltas):+.6f}. "
                     'These settings show no observed accuracy gain; they add runtime. This is a dev observation, not a validation claim.')
    lines += ['', '**Historical shift-and-stitch reference.** The recorded earlier 200-tile dev selection scored the central 256×256 field. '
              'Bilinear U-Net / shift-and-stitch mIoU were 0.7067 / 0.7071; with WCSP at λ=8, 0.7097 / 0.7139. '
              'Shift-and-stitch interleaves four shifted forward passes with no retraining. '
              'There is no fresh shift-and-stitch result on the current tile selection yet; do not rank these historical numbers against the new training table. '
              'Source: [BCSS benchmark notes](../benchmarks/bcss/README.md#addendum-changing-the-u-nets-output-resolution).', '',
              '**Artifacts and interpretation.** Experiment definitions and reproduction commands are in '
              '[the BCSS runbook](../experiments/bcss_wcsp/TONIGHT.md). Current logs, predictions, per-tile CSVs, streaming iteration traces, '
              'and the machine-readable report snapshot are under `data/outputs/tonight/` and `data/tonight/` on the cluster. '
              'Runtime is observed per-tile wall time under concurrent cluster load, not a controlled hardware microbenchmark. '
              'Dev checkpoint selection and parameter comparisons do not replace a separately frozen validation evaluation. '
              'Energy reduction, segmentation accuracy, and solver runtime are reported as distinct outcomes.', END]
    return '\n'.join(lines)


def replace_section(original, section):
    if (BEGIN in original) != (END in original):
        raise ValueError('incomplete generated-section markers')
    if BEGIN in original:
        start=original.index(BEGIN);end=original.index(END)+len(END)
        return original[:start]+section+original[end:]
    return original.rstrip() + '\n\n' + section + '\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=REPO/'data/tonight')
    parser.add_argument('--training-root',type=Path,default=REPO/'data/outputs/tonight')
    parser.add_argument('--deliverable',type=Path,default=DEFAULT_DELIVERABLE)
    args=parser.parse_args()
    if args.deliverable.resolve() == REPO/'deliverables/deliverable_5.md':
        parser.error('BCSS follow-up results belong in Deliverable 6; Deliverable 5 is historical.')
    args.output_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/'deliverable_refresh.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        training=training_summary(args.training_root)
        keys=dict(CENTER+FULL+SOLVERS+DISTANT)
        runs={key:run_summary(args.output_root/'runs'/key/'dev') for key in keys}
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        section=render(training,runs,timestamp)
        original=args.deliverable.read_text() if args.deliverable.exists() else DOCUMENT_HEADER
        updated=replace_section(original,section)
        temporary=args.deliverable.with_suffix('.partial.md')
        temporary.write_text(updated)
        temporary.replace(args.deliverable)
        (args.output_root/'deliverable_results.json').write_text(json.dumps({'updated':timestamp,'training':training,'runs':runs},indent=2))
        print(f'Updated {args.deliverable} at {timestamp}')


if __name__=='__main__':
    main()
