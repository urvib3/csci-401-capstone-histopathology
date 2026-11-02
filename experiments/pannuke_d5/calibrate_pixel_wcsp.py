"""Calibrate on development crops; evaluate a frozen choice on separate crops."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.pannuke_d5.evaluate_fullgrid import evaluate, binary_scores
from experiments.pannuke_d5.run_fullgrid import solve_unified


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def aggregate(rows):
    return {**{k:float(np.mean([r[k] for r in rows])) for k in ('miou','binary_iou','pixel_accuracy','delta_miou')},
            **{k:sum(r[k] for r in rows) for k in ('changed_pixels','wrong_to_correct','correct_to_wrong')},
            'improved_patches':sum(r['delta_miou']>1e-12 for r in rows),
            'worsened_patches':sum(r['delta_miou']< -1e-12 for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['calibrate','evaluate'])
    parser.add_argument('--selection',required=True,type=Path)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--maps',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--frozen-config',type=Path)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    with args.selection.open() as f:
        ids=[int(r['image_index']) for r in csv.DictReader(f)]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate patch indices')
    if args.phase=='calibrate':
        candidates=[('unchanged_hovernet',None)] + [(f'wcsp_{s:g}',s) for s in [0,.1,.25,.5,1,2]]
        config={'probability_head':'original_tp','initialization':'saved_hovernet',
                'background_pairwise_cost':.25,'strength_candidates':[0,.1,.25,.5,1,2],
                'development_indices':ids,'selection_metric':'mean patch biological mIoU; ties prefer unchanged baseline, then lower strength',
                'ground_truth_used_for_optimization':False,'ground_truth_used_for_development_selection':True}
    else:
        if not args.frozen_config:
            parser.error('Evaluation requires --frozen-config')
        config=json.loads(args.frozen_config.read_text())
        if set(ids)&set(config['development_indices']):
            raise ValueError('Evaluation overlaps development')
        strength=config['selected_pairwise_strength']
        # Also evaluate the best actual WCSP if selection chose unchanged HoVer-Net.
        # Both choices are frozen before these evaluation metrics are read.
        actual=config['best_wcsp_strength']
        candidates=[('unchanged_hovernet',None),('frozen_wcsp',actual)]
        config={**config,'frozen_config_sha256':hashlib.sha256(args.frozen_config.read_bytes()).hexdigest(),
                'evaluation_indices':ids,'selected_evaluation_method':'unchanged_hovernet' if strength is None else 'frozen_wcsp'}
    args.output.mkdir(parents=True,exist_ok=True)
    save(args.output/'config.json',config)
    rows=[]
    for index in ids:
        name=f'image_{index:04d}'
        initial=np.load(args.prepared/name/'hovernet_labels.npy',allow_pickle=False)
        with np.load(args.maps/name/'hovernet_probabilities.npz',allow_pickle=False) as data:
            probabilities=(data['type_probability_all'].copy(),)
        folder=args.output/name
        folder.mkdir()
        predictions={}
        for method,strength in candidates:
            if strength is None:
                pred=initial.copy(); report={'optimization_applied':False}
            else:
                result,report=solve_unified(initial,probabilities=probabilities,pairwise_strength=strength)
                pred=result.labels
                report['optimization_applied']=True
            predictions[method]=(pred,report)
            np.save(folder/f'{method}_labels.npy',pred)
        # All predictions for this patch are fixed before its GT is read.
        gt=np.load(args.prepared/name/'ground_truth_labels.npy',allow_pickle=False)
        for method,(pred,report) in predictions.items():
            evaluation=evaluate(initial,pred,gt)
            save(folder/f'{method}.json',{'optimization':report,'evaluation':evaluation})
            rows.append({'image_index':index,'method':method,
                         'miou':evaluation['wcsp']['miou_over_gt_biological_classes'],
                         'binary_iou':binary_scores(pred>0,gt)['nucleus_iou'],
                         'pixel_accuracy':evaluation['wcsp']['pixel_accuracy'],
                         'delta_miou':evaluation['delta_miou_over_gt_biological_classes'],
                         **{k:evaluation[k] for k in ('changed_pixels','wrong_to_correct','correct_to_wrong')}})
        print(f'{args.phase}: completed {name}',flush=True)
    summary={method:aggregate([r for r in rows if r['method']==method]) for method,_ in candidates}
    if args.phase=='calibrate':
        # Stable order resolves exact ties conservatively; no test metrics involved.
        best=max(candidates,key=lambda item:summary[item[0]]['miou'])
        best_wcsp=max(candidates[1:],key=lambda item:summary[item[0]]['miou'])
        config.update(selected_method=best[0],selected_pairwise_strength=best[1],best_wcsp_strength=best_wcsp[1])
        save(args.output/'frozen_config.json',config)
    save(args.output/'summary.json',summary)
    with (args.output/'per_patch.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
