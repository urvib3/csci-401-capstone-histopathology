"""Report a frozen pixel-WCSP evaluation by class and tissue; no model selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS=['Background','Neoplastic','Inflammatory','Connective','Dead','Non-neoplastic epithelial']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--selection',required=True,type=Path)
    parser.add_argument('--frozen-config',required=True,type=Path)
    args=parser.parse_args()
    config=json.loads((args.run/'config.json').read_text())
    assert config['frozen_config_sha256']==hashlib.sha256(args.frozen_config.read_bytes()).hexdigest()
    with args.selection.open() as f:
        selection={int(r['image_index']):r for r in csv.DictReader(f)}
    with (args.run/'per_patch.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['method']=='frozen_wcsp']
    assert len(rows)==len(selection)
    by_tissue={}
    confusions={method:np.zeros((6,6),dtype=np.int64) for method in ['unchanged_hovernet','frozen_wcsp']}
    gt_patch_counts=np.zeros(6,dtype=int)
    local_minima=0
    for row in rows:
        index=int(row['image_index']); name=f'image_{index:04d}'
        by_tissue.setdefault(selection[index]['tissue_type'],[]).append(row)
        gt=np.load(args.prepared/name/'ground_truth_labels.npy',allow_pickle=False)
        gt_patch_counts+=np.array([(gt==k).any() for k in range(6)])
        for method in confusions:
            pred=np.load(args.run/name/f'{method}_labels.npy',allow_pickle=False)
            confusions[method]+=np.bincount((gt.astype(int)*6+pred).ravel(),minlength=36).reshape(6,6)
        report=json.loads((args.run/name/'frozen_wcsp.json').read_text())['optimization']
        assert report['returned_energy']<=report['initial_energy']+1e-8
        local_minima+=report['local_minimum']
    classes=[]
    for k,label in enumerate(LABELS):
        entry={'label':label,'gt_present_patches':int(gt_patch_counts[k]),'gt_pixels':int(confusions['unchanged_hovernet'][k].sum())}
        for method,c in confusions.items():
            union=c[k].sum()+c[:,k].sum()-c[k,k]
            entry[method+'_pooled_iou']=float(c[k,k]/union) if union else None
        classes.append(entry)
    tissues=[]
    for tissue,group in sorted(by_tissue.items()):
        delta=np.array([float(r['delta_miou']) for r in group])
        refined=np.array([float(r['miou']) for r in group])
        tissues.append({'tissue':tissue,'patches':len(group),'hovernet_miou':float((refined-delta).mean()),
                        'wcsp_miou':float(refined.mean()),'delta_miou':float(delta.mean()),
                        'improved':int((delta>1e-12).sum()),'worsened':int((delta< -1e-12).sum())})
    delta=np.array([float(r['delta_miou']) for r in rows])
    rng=np.random.default_rng(401)
    ci=np.quantile(rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1),[.025,.975]).tolist()
    report={'patches':len(rows),'tissue_count':len(tissues),'paired_miou_bootstrap_95_interval':ci,
            'bootstrap_seed':401,'bootstrap_replicates':10000,
            'bootstrap_limitation':'Patch-level resampling; possible patient correlation is not accounted for.',
            'all_energy_nonincreasing':True,'single_pixel_local_minima':int(local_minima),
            'frozen_config_hash_verified':True,'settings_reselected':False,
            'per_class_iou_definition':'IoU from pooled pixel intersections/unions over all evaluation patches, including false positives in GT-absent patches; not the patch-mean headline mIoU.',
            'by_class':classes,'by_tissue':tissues,
            'pooled_confusion_matrices_rows_gt_columns_prediction':{m:c.tolist() for m,c in confusions.items()}}
    (args.run/'detailed_report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,entries in [('per_class',classes),('per_tissue',tissues)]:
        with (args.run/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    summary=json.loads((args.run/'summary.json').read_text())
    lines=['# Frozen pixel WCSP: additional-patch evaluation','',
           f"{len(rows)} additional patches across {len(tissues)} tissue types. One six-label pixel optimization; no binary first stage or nucleus-level WCSP. Full original TP probabilities supply unary costs, saved HoVer-Net labels initialize local search, and neighbor strength remains frozen at 2. No evaluation labels are passed to optimization or used to change settings.",'',
           '| Method | Mean class mIoU | Nucleus/background IoU | Pixel accuracy |',
           '|---|---:|---:|---:|']
    for method,r in summary.items():
        lines.append(f"| {method} | {r['miou']:.6f} | {r['binary_iou']:.6f} | {r['pixel_accuracy']:.6f} |")
    r=summary['frozen_wcsp']
    lines+=['',f"Class mIoU improves on {r['improved_patches']} patches and worsens on {r['worsened_patches']}. Mean change: {r['delta_miou']:+.6f}; paired patch bootstrap 95% interval [{ci[0]:+.6f}, {ci[1]:+.6f}]. It corrects {r['wrong_to_correct']:,} pixels and introduces {r['correct_to_wrong']:,} new errors.",'',
            'Headline class mIoU averages biological classes present in each GT patch, then averages patches, matching prior project reports. Binary nucleus/background IoU is an evaluation of those same six-class predictions after merging biological classes, not a pipeline stage.','',
            '## Per-class IoU','',
            'This table pools pixel intersections/unions across all patches, including false positives where a class is absent. It is not the averaging rule of the headline mIoU.','',
            '| Class | GT-present patches | HoVer-Net IoU | WCSP IoU |','|---|---:|---:|---:|']
    for r in classes:
        a=r['unchanged_hovernet_pooled_iou'];b=r['frozen_wcsp_pooled_iou']
        lines.append(f"| {r['label']} | {r['gt_present_patches']} | {a:.6f} | {b:.6f} |" if a is not None and b is not None else f"| {r['label']} | {r['gt_present_patches']} | unavailable | unavailable |")
    lines+=['','## By tissue','',
            'Small tissue groups are descriptive only; they are not reliable tissue-specific performance estimates.','',
            '| Tissue | Patches | HoVer-Net mIoU | WCSP mIoU | Change |','|---|---:|---:|---:|---:|']
    for r in tissues:
        lines.append(f"| {r['tissue']} | {r['patches']} | {r['hovernet_miou']:.6f} | {r['wcsp_miou']:.6f} | {r['delta_miou']:+.6f} |")
    lines+=['','## Limits and selection','',
            'Only Fold 2 images and saved predictions were available locally. These patches are excluded from prior development, calibration and evaluation selections, but patient-level independence is not verified. Bootstrap intervals resample patches and do not address possible patient correlations. This is additional patch-level validation within the same fold, not independent external validation.','',
            'The development-selected overall method remains unchanged HoVer-Net: all tested WCSP strengths underperformed it on development mIoU. This report separately evaluates the best actual WCSP candidate frozen before the earlier 20-patch evaluation, without using either evaluation set to reselect settings. Equality from retaining the baseline is not evidence of successful WCSP refinement.','',
            f"All {len(rows)} runs have nonincreasing energy; {local_minima} reach a single-pixel local minimum. Local search does not guarantee a global optimum or improved ground-truth accuracy.",'',
            '[Summary](summary.json) · [Per-class data](per_class.csv) · [Per-tissue data](per_tissue.csv) · [Detailed audit](detailed_report.json)','']
    (args.run/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'summary':summary,'paired_miou_bootstrap_95_interval':ci,'local_minima':int(local_minima),'by_class':classes,'by_tissue':tissues},indent=2))


if __name__=='__main__':
    main()
