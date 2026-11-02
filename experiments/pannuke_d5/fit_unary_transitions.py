"""Fit directed PanNuke unary change penalties on patch-disjoint calibration data.

Ground truth is used here to estimate costs, never in the optimization runner.
The 15 development patches, 20 frozen evaluation patches and image 748 are excluded.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.pannuke_d5.run_two_stage_dev import (  # noqa: E402
    ground_truth_from_zip, predicted_inputs, save_json,
)

HERE = Path(__file__).resolve().parent


def costs_from_counts(counts, smoothing=1.0):
    """Higher penalty for transitions less supported than the original class.

    One global normalization keeps mean off-diagonal cost at 1 for comparison
    with the former flat penalty. Diagonal remains zero. No symmetry constraint.
    """
    counts = np.asarray(counts,dtype=float)
    if counts.shape != (5,5) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Expected a finite nonnegative 5x5 calibration count table.')
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('Smoothing must be positive and finite.')
    evidence = counts + smoothing
    raw = np.log1p(np.diag(evidence)[:,None] / evidence)
    np.fill_diagonal(raw,0)
    scale = float(raw[~np.eye(5,dtype=bool)].mean())
    return raw/scale,scale


def calibration_counts(labels,instances,gt):
    """Each usable nucleus contributes unit weight, distributed over its GT types."""
    counts = np.zeros((5,5))
    used = 0
    for instance in np.unique(instances[labels > 0]):
        if instance == 0:
            continue
        mask = (instances == instance) & (labels > 0)
        overlap = mask & (gt > 0)
        # Fit type fidelity from real nuclei, not predominantly false detections.
        if overlap.sum() < .5 * mask.sum():
            continue
        source = int(np.bincount(labels[mask].astype(int),minlength=6)[1:].argmax())
        target = np.bincount(gt[overlap].astype(int),minlength=6)[1:6].astype(float)
        counts[source] += target / target.sum()
        used += 1
    return counts,used


def choose_calibration(catalogue,excluded,count,seed):
    rng = np.random.default_rng(seed)
    pool = [r for r in catalogue if int(r['image_index']) not in excluded
            and r['has_hovernet_predictions'].lower() == 'true'
            and int(r['gt_nucleus_pixels']) > 0]
    if len(pool) < count:
        raise ValueError('Not enough disjoint calibration patches.')
    rng.shuffle(pool)
    # Include rare Dead annotations, then round-robin tissues for broad coverage.
    chosen = [r for r in pool if int(r['gt_dead_pixels']) > 0][:min(10,count)]
    seen = {int(r['image_index']) for r in chosen}
    groups = defaultdict(list)
    for row in pool:
        if int(row['image_index']) not in seen:
            groups[row['tissue_type']].append(row)
    while len(chosen) < count:
        for tissue in sorted(groups):
            if groups[tissue] and len(chosen) < count:
                chosen.append(groups[tissue].pop())
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--patches',type=int,default=100)
    parser.add_argument('--seed',type=int,default=401)
    parser.add_argument('--fold-zip',type=Path,default=ROOT/'benchmarks/pannuke/data/fold_2.zip')
    args = parser.parse_args()
    if args.patches < 1:
        parser.error('--patches must be positive')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('--output must be new or empty')
    excluded = {748}
    for name in ['pannuke_lambda_dev_selection.csv','pannuke_multi_patch_selection.csv']:
        with (HERE/name).open() as f:
            excluded.update(int(r['image_index']) for r in csv.DictReader(f))
    with (HERE/'pannuke_patch_catalog.csv').open() as f:
        chosen = choose_calibration(list(csv.DictReader(f)),excluded,args.patches,args.seed)
    inputs = predicted_inputs(chosen)
    gt = ground_truth_from_zip(args.fold_zip,[int(r['image_index']) for r in chosen])
    total = np.zeros((5,5))
    by_patch = []
    for row in chosen:
        i = int(row['image_index'])
        counts,used = calibration_counts(*inputs[i],gt[i])
        total += counts
        by_patch.append({'image_index':i,'tissue_type':row['tissue_type'],'usable_nuclei':used})
    matrix,scale = costs_from_counts(total)
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'transition_costs.json',matrix.tolist())
    save_json(out/'fit.json',{
        'label_order':[1,2,3,4,5],'seed':args.seed,'patch_count':len(chosen),
        'calibration_patches':by_patch,'excluded_patch_indices':sorted(excluded),
        'usable_nuclei':sum(r['usable_nuclei'] for r in by_patch),
        'nuclei_by_original_class':total.sum(1).tolist(),
        'counts_original_to_gt':total.tolist(),'smoothing':1.0,
        'rule':'offdiagonal log(1 + (C[a,a]+1)/(C[a,b]+1)); diagonal zero; global mean offdiagonal normalized to 1',
        'normalization_divisor':scale,'transition_costs':matrix.tolist(),
        'split_limit':'Patch-disjoint calibration only; patient-level independence is not established.',
    })
    print(json.dumps({'patches':len(chosen),'nuclei':sum(r['usable_nuclei'] for r in by_patch),
                      'source_class_support':total.sum(1).tolist(),'costs':matrix.tolist()},indent=2),flush=True)


if __name__ == '__main__':
    main()
