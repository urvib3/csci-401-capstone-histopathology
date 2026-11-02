"""Compare original pixel TP probabilities and NP/TP fusion, without WCSP."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_fullgrid import binary_scores, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for folder in sorted(args.maps.glob('image_*')):
        source = args.reference / folder.name
        gt = np.load(source / 'ground_truth_labels.npy')
        original = np.load(source / 'hovernet_labels.npy')
        with np.load(folder / 'hovernet_probabilities.npz') as data:
            p = data['nucleus_probability']
            tp = data['type_probability_all']
            conditional = data['type_probability']
        if tp.shape != (6, *gt.shape) or not np.isfinite(tp).all() or not np.allclose(tp.sum(0), 1, atol=1e-5):
            raise ValueError(f'Invalid original TP probabilities: {folder}')
        joint = np.concatenate(((1-p)[None], p[None]*conditional))
        predictions = {'saved_hovernet': original, 'raw_tp_six_channels': tp.argmax(0),
                       'combined_np_tp': joint.argmax(0)}
        row = {'image': folder.name}
        for name, pred in predictions.items():
            row[name] = {**scores(pred, gt), **binary_scores(pred > 0, gt)}
            np.save(folder / f'{name}_labels.npy', pred)
        row['np_head_binary'] = binary_scores(p >= .5, gt)
        rows.append(row)
    if not rows:
        raise ValueError('No exported patches found')
    metrics = ['miou_over_gt_biological_classes', 'pixel_accuracy', 'nucleus_iou', 'nucleus_dice']
    summary = {'patches': len(rows), 'wcsp_used': False, 'clustering_used': False,
               'ground_truth_used_for_prediction': False,
               'metric_definition': 'Unweighted patch mean; biological mIoU averages classes present in each ground-truth patch.',
               'means': {name: {metric: float(np.mean([row[name][metric] for row in rows]))
                                for metric in metrics}
                         for name in predictions}, 'per_patch': rows}
    for name in ('raw_tp_six_channels', 'combined_np_tp'):
        delta = [r[name][metrics[0]] - r['saved_hovernet'][metrics[0]] for r in rows]
        summary[name + '_versus_saved'] = {'improved': sum(d > 1e-12 for d in delta),
                                          'worse': sum(d < -1e-12 for d in delta),
                                          'unchanged': sum(abs(d) <= 1e-12 for d in delta)}
    (args.maps / 'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'per_patch'}, indent=2))


if __name__ == '__main__':
    main()
