"""Fixed-config PanNuke development comparison; no tuning or BCSS changes.

Reads only the existing 15-patch development selection. Reconstructs inputs
using the same centroid mapping and central GT crop as historical preparation.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.pannuke_d5.run_two_stage import run_pipeline  # noqa: E402
from experiments.pannuke_d5.run_fullgrid import make_problem  # noqa: E402
from experiments.pannuke_d5.evaluate_fullgrid import evaluate, binary_scores  # noqa: E402
from wcsp.solve.blocks import BlockCoordinateDescent, BlockLayout  # noqa: E402
from wcsp.solve.icm import ICMSolver  # noqa: E402

HERE = Path(__file__).resolve().parent


def ground_truth_from_zip(path, indices):
    """Stream forward through the NPY member, retaining only selected GT crops."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if n.endswith('/masks.npy') or n == 'masks.npy']
        if len(matches) != 1:
            raise ValueError(f'Expected one masks.npy member, found {matches}')
        with archive.open(matches[0]) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f'Unsupported NPY version {version}')
            if fortran or dtype.hasobject or len(shape) != 4 or shape[1:] != (256,256,6):
                raise ValueError(f'Unsupported PanNuke ground truth: {shape}, {dtype}, fortran={fortran}')
            start = stream.tell()
            size = int(np.prod(shape[1:])) * dtype.itemsize
            for index in sorted(set(indices)):
                if not 0 <= index < shape[0]:
                    raise ValueError(f'Image index {index} outside {shape}')
                stream.seek(start + index * size)
                data = stream.read(size)
                if len(data) != size:
                    raise ValueError('Truncated masks.npy member')
                raw = np.frombuffer(data,dtype=dtype).reshape(shape[1:])[46:210,46:210]
                gt = np.zeros((164,164),dtype=np.uint8)
                for channel in range(5):
                    gt[raw[:,:,channel] > 0] = channel+1
                result[index] = gt
                print(f'Ground truth loaded: image {index}',flush=True)
    return result


def predicted_inputs(selection):
    wanted = {int(r['image_index']) for r in selection}
    directory = ROOT/'benchmarks/pannuke/results/sample_predictions/fold2'
    rows = {i:[] for i in wanted}
    with (directory/'hovernet_fold2_predictions.csv').open() as f:
        for row in csv.DictReader(f):
            i = int(row['image_index'])
            if i in wanted:
                rows[i].append(row)
    result = {}
    for path in sorted(directory.glob('hovernet_instance_maps_*.npz')):
        with np.load(path,allow_pickle=False) as data:
            names = [Path(str(n)).stem for n in data['image_names']]
            targets = [(i,names.index(f'image_00{i}')) for i in wanted if f'image_00{i}' in names]
            if not targets:
                continue
            maps = data['predictions']
            for index,pos in targets:
                instances = maps[pos].copy()
                labels = np.zeros(instances.shape,dtype=np.uint8)
                # Match historical preparation: centroid lookup, ignoring unknown class names.
                classes = {'Neoplastic':1,'Inflammatory':2,'Connective':3,'Dead':4,'Non-neoplastic epithelial':5}
                mapping = {}
                for row in rows[index]:
                    name = row['predicted_type']
                    if name not in classes:
                        continue
                    x,y = round(float(row['centroid_x'])),round(float(row['centroid_y']))
                    if 0 <= y < instances.shape[0] and 0 <= x < instances.shape[1]:
                        instance = int(instances[y,x])
                        if instance:
                            mapping[instance] = classes[name]
                for instance,label in mapping.items():
                    labels[instances == instance] = label
                result[index] = (labels,instances)
                print(f'Prediction inputs loaded: image {index}',flush=True)
    if set(result) != wanted:
        raise ValueError(f'Missing predictions for {sorted(wanted-set(result))}')
    return result


def save_json(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=HERE/'two_stage_config.json')
    parser.add_argument('--fold-zip',type=Path,default=ROOT/'benchmarks/pannuke/data/fold_2.zip')
    parser.add_argument('--reference-results',type=Path,
                        help='Reuse saved GT crops and unchanged unified-reference predictions from a prior run.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be new or empty')
    config = json.loads(args.config.read_text())
    with (HERE/'pannuke_lambda_dev_selection.csv').open() as f:
        selection = list(csv.DictReader(f))
    with (HERE/'pannuke_multi_patch_selection.csv').open() as f:
        frozen = {int(r['image_index']) for r in csv.DictReader(f)}
    indices = [int(r['image_index']) for r in selection]
    if len(indices) != 15 or len(set(indices)) != 15 or set(indices) & frozen:
        raise ValueError('Expected 15 unique development patches disjoint from frozen evaluation')
    inputs = predicted_inputs(selection)
    if args.reference_results:
        gt_maps = {i:np.load(args.reference_results/f'image_{i:04d}'/'ground_truth_labels.npy',allow_pickle=False)
                   for i in indices}
    else:
        gt_maps = ground_truth_from_zip(args.fold_zip,indices)
    with (HERE/'pannuke_lambda_dev_evaluation_summary.csv').open() as f:
        historical = {int(r['image_index']):r for r in csv.DictReader(f) if float(r['lambda']) == 1}
    # Verify preprocessing against saved historical baseline before running experiments.
    for i in indices:
        baseline = evaluate(inputs[i][0],inputs[i][0],gt_maps[i])['hovernet']
        expected = historical[i]
        for actual,key in [(baseline['miou_over_gt_biological_classes'],'hovernet_miou'),
                           (baseline['pixel_accuracy'],'hovernet_full_accuracy')]:
            if not np.isclose(actual,float(expected[key]),rtol=0,atol=1e-12):
                raise ValueError(f'Historical baseline mismatch for image {i}: {key}')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config.json',config)
    summary,failures = [],[]
    for selected in selection:
        i = int(selected['image_index'])
        print(f'Running development patch {i} ({selected["tissue_type"]})',flush=True)
        labels,instances = inputs[i]
        gt = gt_maps[i]
        directory = output/f'image_{i:04d}'
        directory.mkdir()
        np.save(directory/'hovernet_labels.npy',labels)
        np.save(directory/'ground_truth_labels.npy',gt)
        try:
            active,inst,stage1_labels,final,report = run_pipeline(labels,instances,config)
            mask_fixed = dict(config,stage1_boundary_cost=0)
            _,_,_,type_only,type_report = run_pipeline(labels,instances,mask_fixed)
            # Contextual reference: earlier six-label baseline; NOT matched hyperparameters.
            if args.reference_results:
                unified = np.load(args.reference_results/f'image_{i:04d}'/'unified_reference_labels.npy',allow_pickle=False)
            else:
                unified_problem = make_problem(labels,1-np.eye(6),1)
                unified = BlockCoordinateDescent(ICMSolver(),layout=BlockLayout.cross(labels.shape,60),
                                                 max_rounds=config['max_rounds']).solve(unified_problem,labels).labels
            predictions = {'hovernet':labels,'stage1_only':stage1_labels,'stage2_only':type_only,
                           'two_stage':final,'unified_reference':unified}
            save_json(directory/'two_stage_results.json',report)
            save_json(directory/'stage2_only_results.json',type_report)
            np.save(directory/'stage1_active_mask.npy',active)
            np.save(directory/'refined_instance_map.npy',inst)
            for method,pred in predictions.items():
                evaluation = evaluate(labels,pred,gt)
                detection = binary_scores(pred > 0,gt)
                np.save(directory/f'{method}_labels.npy',pred)
                save_json(directory/f'{method}_evaluation.json',evaluation)
                summary.append({'image_index':i,'tissue_type':selected['tissue_type'],'method':method,
                                'miou':evaluation['wcsp']['miou_over_gt_biological_classes'],
                                'delta_miou':evaluation['delta_miou_over_gt_biological_classes'],
                                'pixel_accuracy':evaluation['wcsp']['pixel_accuracy'],
                                'binary_iou':detection['nucleus_iou'],'binary_dice':detection['nucleus_dice'],
                                'changed_pixels':evaluation['changed_pixels'],
                                'wrong_to_correct':evaluation['wrong_to_correct'],
                                'correct_to_wrong':evaluation['correct_to_wrong']})
        except ValueError as exc:
            failures.append({'image_index':i,'error':str(exc)})
            print(f'Patch {i} failed: {exc}',flush=True)
        save_json(output/'failures.json',failures)
        if summary:
            with (output/'per_patch.csv').open('w',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=list(summary[0]))
                writer.writeheader();writer.writerows(summary)
    aggregate = {'selection':'pannuke_lambda_dev_selection.csv','requested_patches':15,
                 'successful_patches':len({r['image_index'] for r in summary}),
                 'historical_baselines_verified':True,'tuned_on_this_run':False,
                 'probability_maps_used':False,'failures':failures,'methods':{}}
    for method in ['hovernet','stage1_only','stage2_only','two_stage','unified_reference']:
        rows = [r for r in summary if r['method'] == method]
        if not rows:
            continue
        aggregate['methods'][method] = {
            'patches':len(rows),
            **{f'mean_{k}':float(np.mean([r[k] for r in rows]))
               for k in ['miou','delta_miou','pixel_accuracy','binary_iou','binary_dice']},
            **{k:sum(r[k] for r in rows) for k in ['changed_pixels','wrong_to_correct','correct_to_wrong']},
            'improved_patches':sum(r['delta_miou'] > 1e-12 for r in rows),
            'worsened_patches':sum(r['delta_miou'] < -1e-12 for r in rows),
            'unchanged_patches':sum(abs(r['delta_miou']) <= 1e-12 for r in rows)}
    save_json(output/'summary.json',aggregate)
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__ == '__main__':
    main()
