"""PanNuke-only binary mask WCSP followed by nucleus-level type WCSP.

No ground truth is read. This experiment uses the shared solver without
changing its five-class defaults or any BCSS implementation.
"""

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wcsp.model.neighborhood import distance_weights, forward_offsets, pair_slices  # noqa: E402
from wcsp.model.problem import GridProblem  # noqa: E402
from wcsp.model.tables import HANDWRITTEN  # noqa: E402
from wcsp.solve.blocks import BlockCoordinateDescent, BlockLayout  # noqa: E402
from wcsp.solve.energy import energy, local_costs  # noqa: E402
from wcsp.solve.icm import ICMSolver  # noqa: E402


def transition_matrix(values, size):
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f'Transition costs must be a finite nonnegative {size}x{size} matrix.')
    if np.any(matrix.diagonal() != 0):
        raise ValueError('Transition matrix diagonal must be zero.')
    return matrix


def nonnegative(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return float(value)


def probability_map(values, shape, channels=None):
    values = np.asarray(values, dtype=float)
    expected = shape if channels is None else (channels, *shape)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f'Probability array must be finite with shape {expected}.')
    if ((values < 0) | (values > 1)).any():
        raise ValueError('Probabilities must be in [0, 1].')
    if channels is not None and not np.allclose(values.sum(axis=0), 1, atol=1e-5):
        raise ValueError('Five type probabilities must sum to one at every pixel.')
    return values


def binary_problem(initial, transitions, boundary_cost, nucleus_probability=None):
    initial = np.asarray(initial, dtype=bool)
    transitions = transition_matrix(transitions, 2)
    unary = transitions[initial.astype(int)].copy()
    if nucleus_probability is not None:
        p = probability_map(nucleus_probability, initial.shape)
        unary += -np.log(np.clip(np.stack([1-p, p], axis=-1), 1e-8, 1))
    offsets = forward_offsets()
    return GridProblem(
        unary=unary, offsets=offsets, weights=distance_weights(offsets, 1),
        table=nonnegative(boundary_cost, 'stage1_boundary_cost') * (1-np.eye(2)),
        active=np.ones(initial.shape, dtype=bool),
    )


def neighbors(y, x, shape):
    for yy, xx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
        if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
            yield yy, xx


def reconstruct_instances(active, original_instances):
    """Keep distinct HoVer-Net markers; expand them only inside the new mask.

    Split disconnected surviving pieces of a marker. Unseeded foreground
    components become new candidate instances. This is a deterministic
    four-connected geodesic partition, not a new learned instance detector.
    """
    active = np.asarray(active, dtype=bool)
    original_instances = np.asarray(original_instances)
    if original_instances.shape != active.shape:
        raise ValueError('Instance map must match the active mask.')
    if not np.issubdtype(original_instances.dtype, np.integer) or (original_instances < 0).any():
        raise ValueError('Instance IDs must be nonnegative integers.')
    result = np.zeros(active.shape, dtype=np.int32)
    count = 0
    for y, x in np.argwhere(active & (original_instances > 0)):
        if result[y, x]:
            continue
        count += 1
        marker = original_instances[y, x]
        result[y, x] = count
        queue = deque([(y,x)])
        while queue:
            yy,xx = queue.popleft()
            for a,b in neighbors(yy,xx,active.shape):
                if active[a,b] and not result[a,b] and original_instances[a,b] == marker:
                    result[a,b] = count
                    queue.append((a,b))
    # Simultaneous breadth-first expansion keeps touching original nuclei distinct.
    queue = deque(map(tuple, np.argwhere(result > 0)))
    while queue:
        y,x = queue.popleft()
        for a,b in neighbors(y,x,active.shape):
            if active[a,b] and not result[a,b]:
                result[a,b] = result[y,x]
                queue.append((a,b))
    for y,x in np.argwhere(active & (result == 0)):
        if result[y,x]:
            continue
        count += 1
        result[y,x] = count
        queue = deque([(y,x)])
        while queue:
            yy,xx = queue.popleft()
            for a,b in neighbors(yy,xx,active.shape):
                if active[a,b] and not result[a,b]:
                    result[a,b] = count
                    queue.append((a,b))
    return result


def nucleus_unaries(instances, original_labels, transitions, type_probability=None):
    transitions = transition_matrix(transitions, 5)
    if instances.shape != original_labels.shape:
        raise ValueError('Instance and label maps must have matching shapes.')
    if type_probability is not None:
        type_probability = probability_map(type_probability, instances.shape, 5)
    count = int(instances.max(initial=0))
    unary = np.zeros((count,5))
    initial = np.zeros(count,dtype=int)
    for i in range(1,count+1):
        mask = instances == i
        retained = original_labels[mask & (original_labels > 0)].astype(int)-1
        if retained.size:
            # Mean costs keep the cost scale comparable across different nucleus sizes.
            unary[i-1] = transitions[retained].mean(axis=0)
            initial[i-1] = np.bincount(retained,minlength=5).argmax()
        elif type_probability is None:
            raise ValueError(f'New nucleus {i} has no saved type evidence. Supply --type-probabilities.')
        if type_probability is not None:
            unary[i-1] += -np.log(np.clip(type_probability[:,mask],1e-8,1)).mean(axis=1)
            if not retained.size:
                if np.ptp(unary[i-1]) < 1e-12:
                    raise ValueError(f'New nucleus {i} has uniform type evidence; cannot choose its type.')
                initial[i-1] = unary[i-1].argmin()
    return unary, initial


def nucleus_edges(instances, radius):
    """One edge per nucleus pair within radius, weighted by inverse boundary distance."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Nucleus neighbor radius must be finite and nonnegative.')
    pairs = {}
    reach = min(int(np.ceil(radius)), max(instances.shape)-1)
    for dy in range(0,reach+1):
        for dx in range(-reach,reach+1):
            distance = float(np.hypot(dy,dx))
            if distance == 0 or distance > radius or (dy == 0 and dx <= 0):
                continue
            if dy >= instances.shape[0] or abs(dx) >= instances.shape[1]:
                continue
            p,q = pair_slices(instances.shape, (dy,dx))
            a,b = instances[p],instances[q]
            keep = (a > 0) & (b > 0) & (a != b)
            if not keep.any():
                continue
            ordered = np.sort(np.stack([a[keep],b[keep]],axis=1),axis=1)
            for left,right in np.unique(ordered,axis=0):
                key = (int(left)-1,int(right)-1)
                pairs[key] = min(pairs.get(key,float('inf')),distance)
    return [(a,b,1/d) for (a,b),d in sorted(pairs.items())]


def graph_energy(labels, unary, edges, strength):
    return float(unary[np.arange(len(labels)),labels].sum() + strength * sum(
        weight * HANDWRITTEN[labels[a],labels[b]] for a,b,weight in edges))


def solve_types(unary, initial, edges, strength, max_rounds):
    strength = nonnegative(strength,'stage2_pairwise_strength')
    labels = initial.copy()
    adjacent = [[] for _ in labels]
    for a,b,weight in edges:
        adjacent[a].append((b,weight))
        adjacent[b].append((a,weight))
    trace = [graph_energy(labels,unary,edges,strength)]
    def costs(i):
        values = unary[i].copy()
        for j,weight in adjacent[i]:
            values += strength * weight * HANDWRITTEN[:,labels[j]]
        return values
    for _ in range(max_rounds):
        changed = 0
        for i in range(len(labels)):
            values = costs(i)
            best = values.argmin()
            if values[best] < values[labels[i]]-1e-12:
                labels[i] = best
                changed += 1
        trace.append(graph_energy(labels,unary,edges,strength))
        if not changed:
            break
    local_minimum = all(costs(i).min() >= costs(i)[labels[i]]-1e-9 for i in range(len(labels)))
    return labels,trace,local_minimum


def run_pipeline(labels, original_instances, config, nucleus_probability=None, type_probability=None):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype,np.integer) or not np.isin(labels,range(6)).all():
        raise ValueError('Original labels must be a 2-D integer map in 0..5.')
    if original_instances.shape != labels.shape:
        raise ValueError('Instance and label maps must match.')
    if ((labels > 0) & (original_instances == 0)).any():
        raise ValueError('Each original typed nucleus pixel must have an instance ID.')
    max_rounds = config['max_rounds']
    if not isinstance(max_rounds,int) or isinstance(max_rounds,bool) or max_rounds < 1:
        raise ValueError('max_rounds must be a positive integer.')
    initial_mask = labels > 0
    problem = binary_problem(initial_mask,config['stage1_transitions'],
                             config['stage1_boundary_cost'],nucleus_probability)
    shape = initial_mask.shape
    layout = BlockLayout.cross(shape,block=60) if min(shape) >= 123 else BlockLayout.whole(shape)
    binary = BlockCoordinateDescent(ICMSolver(),layout=layout,max_rounds=max_rounds).solve(
        problem,initial_mask.astype(int))
    active = binary.labels.astype(bool)
    # Untyped HoVer-Net instances cannot become classification evidence on their own.
    markers = np.where(initial_mask,original_instances,0)
    instances = reconstruct_instances(active,markers)
    unary,type_initial = nucleus_unaries(instances,labels,config['stage2_transitions'],type_probability)
    edges = nucleus_edges(instances,config['stage2_neighbor_radius'])
    types,trace,local_min = solve_types(unary,type_initial,edges,
                                      config['stage2_pairwise_strength'],max_rounds)
    final = np.zeros(labels.shape,dtype=np.uint8)
    stage2_initial = np.zeros_like(final)
    rows = []
    for i,label in enumerate(types,1):
        mask = instances == i
        final[mask] = int(label)+1
        stage2_initial[mask] = int(type_initial[i-1])+1
        y,x = np.nonzero(mask)
        rows.append({'nucleus_id':i,'area_px':int(mask.sum()),
                     'centroid_x':float(x.mean()),'centroid_y':float(y.mean()),
                     'initial_type_id':int(type_initial[i-1])+1,'refined_type_id':int(label)+1,
                     'unary_cost_vector':unary[i-1].tolist(),
                     'original_instance_ids':sorted(int(v) for v in np.unique(markers[mask]) if v > 0)})
    local = local_costs(problem,binary.labels)
    chosen = np.take_along_axis(local,binary.labels[...,None],-1)[...,0]
    report = {
        'mode':'two_stage_nucleus_graph','ground_truth_used':False,'config':config,
        'evidence':{'nucleus_probability_used':nucleus_probability is not None,
                    'type_probability_used':type_probability is not None,
                    'hard_label_fallback':nucleus_probability is None or type_probability is None},
        'stage1':{'variable_count':int(labels.size),'domain':['background','nucleus'],
                  'initial_energy':energy(problem,initial_mask.astype(int)),
                  'returned_energy':binary.energy,'background_to_active':int((~initial_mask & active).sum()),
                  'active_to_background':int((initial_mask & ~active).sum()),
                  'local_minimum':bool((local.min(-1) >= chosen-1e-9).all())},
        'stage2':{'variable_count':len(types),'domain':[1,2,3,4,5],'edge_count':len(edges),
                  'initial_energy':trace[0],'returned_energy':trace[-1],'energy_trace':trace,
                  'nuclei_retyped':int((types != type_initial).sum()),'local_minimum':local_min,
                  'new_unseeded_instances':sum(not row['original_instance_ids'] for row in rows)},
        'nuclei':rows,
    }
    return active,instances,stage2_initial,final,report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('two_stage_config.json'))
    parser.add_argument('--nucleus-probabilities',type=Path,help='Aligned (H,W) P(nucleus) array.')
    parser.add_argument('--type-probabilities',type=Path,help='Aligned (5,H,W) conditional type probabilities, classes 1..5.')
    args = parser.parse_args()
    source,output = args.source.resolve(),args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be new or empty; historical results will not be overwritten')
    load = lambda p: np.load(p,allow_pickle=False)
    labels = load(source/'hovernet_labels_164.npy')
    instances = load(source/'hovernet_instance_map_164.npy')
    config = json.loads(args.config.read_text())
    p = load(args.nucleus_probabilities) if args.nucleus_probabilities else None
    t = load(args.type_probabilities) if args.type_probabilities else None
    active,instances,type_initial,refined,report = run_pipeline(labels,instances,config,p,t)
    report['source'] = str(source)
    report['probability_sources'] = {
        'nucleus':str(args.nucleus_probabilities.resolve()) if args.nucleus_probabilities else None,
        'type':str(args.type_probabilities.resolve()) if args.type_probabilities else None,
    }
    output.mkdir(parents=True,exist_ok=True)
    for name,values in [('stage1_active_mask',active),('refined_instance_map',instances),
                        ('stage2_initial_labels',type_initial),('refined_labels',refined)]:
        np.save(output/f'{name}.npy',values)
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'nuclei.csv').open('w',newline='') as f:
        columns=['nucleus_id','area_px','centroid_x','centroid_y','initial_type_id',
                 'refined_type_id','unary_cost_vector','original_instance_ids']
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(report['nuclei'])
    print(json.dumps({k:report[k] for k in ['evidence','stage1','stage2']},indent=2))


if __name__ == '__main__':
    main()
