"""Evaluate probability-based unified WCSP, or reproduce the old transition-cost comparison."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.pannuke_d5.run_fullgrid import (  # noqa: E402
    DEFAULT_TRANSITIONS, LABELS, probability_unary, solve_unified, validate_transitions,
)
from experiments.pannuke_d5.evaluate_fullgrid import evaluate, binary_scores  # noqa: E402

HERE = Path(__file__).resolve().parent


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepared-results',type=Path,default=HERE/'two_stage_dev_trial',
                        help='Existing cached HoVer-Net labels and GT crops; no stage outputs used as input.')
    parser.add_argument('--transition-costs',type=Path,default=DEFAULT_TRANSITIONS)
    parser.add_argument('--unary-mode',choices=['probabilities','transitions'],default='probabilities')
    parser.add_argument('--probability-dir',type=Path,
                        help='Export root containing image_NNNN/hovernet_probabilities.npz.')
    parser.add_argument('--probability-head',choices=['combined_np_tp','original_tp'],default='combined_np_tp',
                        help='original_tp uses all six original TP softmax channels, without NP fusion.')
    parser.add_argument('--initialization',choices=['probability_argmax','hovernet'],default='hovernet',
                        help='Keep the better-tested initialization by default; probability_argmax tests initialization solely from pixel evidence.')
    parser.add_argument('--skip-type-prior',action='store_true',
                        help='Run only probability evidence + spatial costs, without the calibrated type-prior comparison.')
    parser.add_argument('--background-pairwise-cost',type=float,default=.25)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be new or empty')
    uses_transition_table = args.unary_mode=='transitions' or not args.skip_type_prior
    matrix=(validate_transitions(json.loads(args.transition_costs.read_text()))
            if uses_transition_table else np.zeros((6,6)))
    flat=matrix.copy()
    flat[1:,1:]=1-np.eye(5)
    with (HERE/'pannuke_lambda_dev_selection.csv').open() as f:
        selection=list(csv.DictReader(f))
    with (HERE/'pannuke_multi_patch_selection.csv').open() as f:
        reserved={int(r['image_index']) for r in csv.DictReader(f)}
    ids={int(r['image_index']) for r in selection}
    if len(selection)!=15 or len(ids)!=15 or ids & reserved:
        raise ValueError('Expected 15 unique development patches disjoint from evaluation.')
    if args.unary_mode=='probabilities':
        if not args.probability_dir:
            parser.error('--probability-dir is required in probability mode; export actual HoVer-Net maps first.')
        missing=[i for i in ids if not (args.probability_dir/f'image_{i:04d}'/'hovernet_probabilities.npz').is_file()]
        if missing:
            parser.error(f'Missing actual probability maps for images {sorted(missing)}')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config.json',{'unary_mode':args.unary_mode,
                                  'probability_head':args.probability_head,
                                  'transition_costs':matrix.tolist() if uses_transition_table else None,
                                  'initialization':args.initialization if args.unary_mode=='probabilities' else 'hovernet',
                                  'skip_type_prior':args.skip_type_prior,
                                  'probability_dir':str(args.probability_dir.resolve()) if args.probability_dir else None,
                                  'background_pairwise_cost':args.background_pairwise_cost,
                                  'transition_source':str(args.transition_costs.resolve()) if uses_transition_table else None,
                                  'flat_control':('same background costs and graph; only biological offdiagonal unaries = 1'
                                                  if args.unary_mode=='transitions' else None),
                                  'unary_evidence':('full per-pixel HoVer-Net probabilities; no class-transition prior'
                                                    if not uses_transition_table else 'see unary mode and transition table')})
    rows=[]
    for selected in selection:
        i=int(selected['image_index'])
        print(f'Unified development patch {i}',flush=True)
        cached=args.prepared_results/f'image_{i:04d}'
        initial=np.load(cached/'hovernet_labels.npy',allow_pickle=False)
        folder=output/f'image_{i:04d}'
        folder.mkdir()
        predictions={'hovernet':initial}
        runs=[]
        probabilities=None
        optimization_initial=initial
        if args.unary_mode=='probabilities':
            with np.load(args.probability_dir/f'image_{i:04d}'/'hovernet_probabilities.npz',allow_pickle=False) as data:
                probabilities=((data['type_probability_all'],) if args.probability_head=='original_tp'
                               else (data['nucleus_probability'],data['type_probability']))
            predictions['probability_argmax'] = probability_unary(*probabilities).argmin(axis=-1)
            if args.initialization=='probability_argmax':
                optimization_initial=predictions['probability_argmax']
            methods=[('unified_probabilities',matrix,False),('probabilities_with_type_prior',matrix,True)]
            if args.skip_type_prior:
                methods=methods[:1]
            final_method='unified_probabilities'
        else:
            methods=[('flat_type_control',flat,False),('unified_class_costs',matrix,False)]
            final_method='unified_class_costs'
        for method,costs,prior in methods:
            result,report=solve_unified(optimization_initial,costs,args.background_pairwise_cost,
                                        probabilities=probabilities,add_transition_prior=prior)
            runs.append(dict(report,setting=method))
            predictions[method]=result.labels
            np.save(folder/f'{method}_labels.npy',result.labels)
        # Ground truth is loaded after both optimizations and never passed to them.
        gt=np.load(cached/'ground_truth_labels.npy',allow_pickle=False)
        np.save(folder/'hovernet_labels_164.npy',initial)
        np.save(folder/'optimization_initial_labels.npy',optimization_initial)
        np.save(folder/'pannuke_gt_labels_164.npy',gt)
        np.save(folder/'refined_labels.npy',predictions[final_method])
        np.save(folder/'active_mask.npy',predictions[final_method]>0)
        save_json(folder/'results.json',{
            'source':str(folder),'ground_truth_used':False,'mode':'all_pixels_six_classes',
            'unary_mode':args.unary_mode,
            'probability_head':args.probability_head,
            'initialization':args.initialization if probabilities is not None else 'hovernet',
            'clustering_used':False,'nucleus_averaging_used':False,
            'label_dictionary':dict(enumerate(LABELS)),'domain':list(range(6)),
            'variable_count':int(initial.size),'unary_transition_costs':matrix.tolist() if probabilities is None else None,
            'optional_type_transition_prior':matrix[1:,1:].tolist() if probabilities is not None and not args.skip_type_prior else None,
            'background_pairwise_cost':args.background_pairwise_cost,'runs':runs,
        })
        for method,pred in predictions.items():
            scores=evaluate(initial,pred,gt)
            detection=binary_scores(pred>0,gt)
            save_json(folder/f'{method}_evaluation.json',dict(scores,binary_segmentation=detection))
            rows.append({'image_index':i,'tissue_type':selected['tissue_type'],'method':method,
                         'miou':scores['wcsp']['miou_over_gt_biological_classes'],
                         'delta_miou':scores['delta_miou_over_gt_biological_classes'],
                         'binary_iou':detection['nucleus_iou'],
                         'pixel_accuracy':scores['wcsp']['pixel_accuracy'],
                         'changed_pixels':scores['changed_pixels'],
                         'wrong_to_correct':scores['wrong_to_correct'],
                         'correct_to_wrong':scores['correct_to_wrong']})
    with (output/'per_patch.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary={'mode':'one_stage_six_class_pixel_wcsp','unary_mode':args.unary_mode,
             'probability_head':args.probability_head,
             'initialization':args.initialization if args.unary_mode=='probabilities' else 'hovernet',
             'type_prior_included':args.unary_mode=='probabilities' and not args.skip_type_prior,
             'patches':15,'costs_tuned_on_dev':False,'methods':{}}
    for method in predictions:
        selected=[r for r in rows if r['method']==method]
        summary['methods'][method]={
            **{f'mean_{key}':float(np.mean([r[key] for r in selected]))
               for key in ['miou','delta_miou','binary_iou','pixel_accuracy']},
            **{key:sum(r[key] for r in selected)
               for key in ['changed_pixels','wrong_to_correct','correct_to_wrong']},
            'improved_patches':sum(r['delta_miou']>1e-12 for r in selected),
            'worsened_patches':sum(r['delta_miou']< -1e-12 for r in selected),
            'unchanged_patches':sum(abs(r['delta_miou'])<=1e-12 for r in selected)}
    save_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
