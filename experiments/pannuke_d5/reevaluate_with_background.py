"""Re-evaluate saved six-class predictions including background; no optimization."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

LABELS=['Background','Neoplastic','Inflammatory','Connective','Dead','Non-neoplastic epithelial']


def confusion(gt, pred):
    if gt.shape != pred.shape or not np.isin(gt,range(6)).all() or not np.isin(pred,range(6)).all():
        raise ValueError('Aligned label maps in 0..5 are required')
    return np.bincount((gt.astype(int)*6+pred).ravel(),minlength=36).reshape(6,6)


def ious(matrix):
    intersections=np.diag(matrix)
    unions=matrix.sum(0)+matrix.sum(1)-intersections
    return np.divide(intersections,unions,out=np.full(6,np.nan),where=unions>0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--refined-name',default='frozen_wcsp')
    args=parser.parse_args()
    totals={m:np.zeros((6,6),dtype=np.int64) for m in ['hovernet','wcsp']}
    rows=[]
    for folder in sorted(args.run.glob('image_*')):
        baseline=folder/'unchanged_hovernet_labels.npy'
        if not baseline.exists():
            continue
        gt=np.load(args.prepared/folder.name/'ground_truth_labels.npy',allow_pickle=False)
        preds={'hovernet':np.load(baseline,allow_pickle=False),
               'wcsp':np.load(folder/f'{args.refined_name}_labels.npy',allow_pickle=False)}
        row={'image':folder.name}
        # Identical GT-present class set for both methods, including background.
        present=np.bincount(gt.ravel(),minlength=6)>0
        for method,pred in preds.items():
            matrix=confusion(gt,pred);totals[method]+=matrix
            values=ious(matrix)
            row[method+'_patch_miou_gt_classes_with_background']=float(values[present].mean())
            row[method+'_missed_nucleus_pixels']=int(((gt>0)&(pred==0)).sum())
            row[method+'_false_positive_nucleus_pixels']=int(((gt==0)&(pred>0)).sum())
        row['delta_patch_miou_with_background']=(row['wcsp_patch_miou_gt_classes_with_background']-
                                                 row['hovernet_patch_miou_gt_classes_with_background'])
        rows.append(row)
    if not rows:
        raise ValueError('No saved prediction pairs found')
    report={'patches':len(rows),'optimization_rerun':False,'settings_reselected':False,
            'labels':LABELS,
            'pooled_metric_definition':'Accumulate pixel confusion matrix across patches, then average IoU equally across all six labels including background. Any label with zero dataset union is undefined and omitted.',
            'patch_metric_definition':'For each patch average IoU over its GT-present classes including background, using the same class set for both methods; then average patches. Retained for comparison with earlier patch-mean metrics. Classes absent in GT are not separately averaged by this diagnostic.',
            'methods':{}}
    for method,matrix in totals.items():
        values=ious(matrix)
        report['methods'][method]={
            'pooled_six_class_miou_including_background':float(np.nanmean(values)),
            'pooled_class_iou':{label:float(v) if np.isfinite(v) else None for label,v in zip(LABELS,values)},
            'pooled_pixel_accuracy':float(matrix.trace()/matrix.sum()),
            'mean_patch_miou_gt_classes_including_background':float(np.mean([r[method+'_patch_miou_gt_classes_with_background'] for r in rows])),
            'missed_nucleus_pixels':sum(r[method+'_missed_nucleus_pixels'] for r in rows),
            'false_positive_nucleus_pixels':sum(r[method+'_false_positive_nucleus_pixels'] for r in rows),
            'confusion_matrix_rows_gt_columns_prediction':matrix.tolist()}
    delta=np.array([r['delta_patch_miou_with_background'] for r in rows])
    report['patch_comparison_with_background']={'improved':int((delta>1e-12).sum()),
                                               'worsened':int((delta< -1e-12).sum()),
                                               'unchanged':int((abs(delta)<=1e-12).sum())}
    (args.run/'background_reassessment.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.run/'background_reassessment_per_patch.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Metrics re-evaluated including background','',
           f"Re-evaluation of the same {len(rows)} saved prediction pairs. No optimization, label changes, parameter tuning or per-patch ground-truth acceptance gate was performed.",'',
           '| Method | Pooled six-class mIoU including background | Mean patch mIoU including background (GT-present classes) | Pixel accuracy |',
           '|---|---:|---:|---:|']
    for method,r in report['methods'].items():
        lines.append(f"| {method} | {r['pooled_six_class_miou_including_background']:.6f} | {r['mean_patch_miou_gt_classes_including_background']:.6f} | {r['pooled_pixel_accuracy']:.6f} |")
    lines+=['',report['pooled_metric_definition'],'',report['patch_metric_definition'],'',
            '## Per-class pooled IoU','', '| Class | HoVer-Net | WCSP |','|---|---:|---:|']
    for label in LABELS:
        a=report['methods']['hovernet']['pooled_class_iou'][label]
        b=report['methods']['wcsp']['pooled_class_iou'][label]
        lines.append(f'| {label} | {a:.6f} | {b:.6f} |' if a is not None and b is not None else f'| {label} | {a} | {b} |')
    comparison=report['patch_comparison_with_background']
    lines+=['',f"Using the background-inclusive patch metric: {comparison['improved']} patches improve, {comparison['worsened']} worsen and {comparison['unchanged']} are unchanged.",'',
            'The earlier headline metric averaged only GT-present biological classes. Background was already a possible WCSP label and background errors already affected biological unions and pixel accuracy; its own IoU was excluded from that headline average. The new pooled six-class metric also changes the aggregation from a patch mean to dataset-pooled intersections/unions. Both changes are shown explicitly; the new pooled result must not be compared directly to the old patch mean as if the model improved. A dataset-level mean gain does not establish non-degradation for every patch.','',
            'Low WCSP energy guarantees neither higher IoU nor higher ground-truth accuracy. Rejecting changes based on each evaluation patch\'s ground truth would leak evaluation labels. The unchanged HoVer-Net baseline is reported separately from actual WCSP results.','',
            '[Full metrics](background_reassessment.json) · [Paired patch metrics](background_reassessment_per_patch.csv)','']
    (args.run/'BACKGROUND_REASSESSMENT.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k not in ['labels','pooled_metric_definition','patch_metric_definition']},indent=2))


if __name__=='__main__':
    main()
