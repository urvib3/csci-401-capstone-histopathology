"""Evaluate the unified or two-stage PanNuke experiment separately from optimization."""

import argparse
import json
from pathlib import Path

import numpy as np


def scores(pred, gt):
    iou = []
    for label in range(6):
        union = ((pred == label) | (gt == label)).sum()
        iou.append(float(((pred == label) & (gt == label)).sum() / union)
                   if union else None)
    biological = [iou[k] for k in range(1, 6) if (gt == k).any()]
    return {
        'pixel_accuracy': float((pred == gt).mean()),
        'class_iou_labels_0_to_5': iou,
        'miou_over_gt_biological_classes': float(np.mean(biological)) if biological else None,
        'miou_over_gt_classes_including_background': float(np.mean(
            [iou[k] for k in range(6) if (gt == k).any()])),
        'missed_nucleus_pixels': int(((gt > 0) & (pred == 0)).sum()),
        'false_positive_nucleus_pixels': int(((gt == 0) & (pred > 0)).sum()),
    }


def evaluate(initial, refined, gt):
    if initial.shape != refined.shape or initial.shape != gt.shape:
        raise ValueError('Predictions and ground truth must have matching shapes.')
    for array in (initial, refined, gt):
        if not np.issubdtype(array.dtype, np.integer) or not np.isin(array, range(6)).all():
            raise ValueError('All maps must contain integer labels 0..5.')
    before, after = scores(initial, gt), scores(refined, gt)
    changed = initial != refined
    was, now = initial == gt, refined == gt
    transitions = {}
    for a, b in zip(initial[changed], refined[changed]):
        key = f'{a}->{b}'
        transitions[key] = transitions.get(key, 0) + 1
    return {
        'hovernet': before, 'wcsp': after,
        'delta_pixel_accuracy': after['pixel_accuracy'] - before['pixel_accuracy'],
        'delta_miou_over_gt_biological_classes': (
            after['miou_over_gt_biological_classes'] - before['miou_over_gt_biological_classes']
            if before['miou_over_gt_biological_classes'] is not None else None),
        'changed_pixels': int(changed.sum()),
        # All pixels count, including genuine background corrections and new errors.
        'wrong_to_correct': int((changed & ~was & now).sum()),
        'correct_to_wrong': int((changed & was & ~now).sum()),
        'wrong_to_different_wrong': int((changed & ~was & ~now).sum()),
        'changes_by_transition': transitions,
    }


def binary_scores(mask, gt):
    target = gt > 0
    intersection = int((mask & target).sum())
    union = int((mask | target).sum())
    total = int(mask.sum() + target.sum())
    return {'nucleus_iou': intersection / union if union else 1.0,
            'nucleus_dice': 2 * intersection / total if total else 1.0,
            'missed_nucleus_pixels': int((target & ~mask).sum()),
            'false_positive_nucleus_pixels': int((~target & mask).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    metadata = json.loads((run / 'results.json').read_text())
    if metadata.get('mode') not in ('all_pixels_six_classes', 'two_stage_nucleus_graph') or metadata.get('ground_truth_used') is not False:
        parser.error('Expected a ground-truth-free unified or two-stage run.')
    source = Path(metadata['source'])
    initial = np.load(source / 'hovernet_labels_164.npy', allow_pickle=False)
    refined = np.load(run / 'refined_labels.npy', allow_pickle=False)
    gt = np.load(source / 'pannuke_gt_labels_164.npy', allow_pickle=False)
    report = evaluate(initial, refined, gt)
    report['binary_segmentation'] = {
        'hovernet': binary_scores(initial > 0, gt),
        'refined': binary_scores(refined > 0, gt),
    }
    if metadata['mode'] == 'two_stage_nucleus_graph':
        stage2_initial = np.load(run / 'stage2_initial_labels.npy', allow_pickle=False)
        report['stage2_only'] = evaluate(stage2_initial, refined, gt)
        report['stage2_mask_unchanged'] = bool(np.array_equal(stage2_initial > 0, refined > 0))
    (run / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
