"""PanNuke experiment: all pixels, six labels, confidence-based unary costs.

The historical nucleus-only solver and its saved results are unchanged.
Ground truth is read only by evaluate_fullgrid.py, never by this script.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wcsp.model.neighborhood import distance_weights, forward_offsets  # noqa: E402
from wcsp.model.problem import GridProblem  # noqa: E402
from wcsp.model.tables import HANDWRITTEN  # noqa: E402
from wcsp.solve.blocks import BlockCoordinateDescent, BlockLayout  # noqa: E402
from wcsp.solve.energy import energy, local_costs  # noqa: E402
from wcsp.solve.icm import ICMSolver  # noqa: E402

LABELS = ['Background', 'Neoplastic', 'Inflammatory', 'Connective',
          'Dead', 'Non-neoplastic epithelial']
DEFAULT_TRANSITIONS = Path(__file__).with_name('unified_transition_costs.json')


def validate_transitions(costs):
    costs = np.asarray(costs, dtype=float)
    if costs.shape != (6, 6):
        raise ValueError('Unary transition costs must be a 6 x 6 matrix.')
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Unary transition costs must be finite and non-negative.')
    if np.any(np.diag(costs) != 0):
        raise ValueError('Keeping the original class must cost zero (diagonal = 0).')
    return costs


def make_problem(initial, transitions, background_cost, pairwise=True):
    initial = np.asarray(initial)
    if initial.ndim != 2 or not np.issubdtype(initial.dtype, np.integer):
        raise ValueError('Initial labels must be a 2-D integer array.')
    if not np.isin(initial, np.arange(6)).all():
        raise ValueError('Initial labels must be in 0..5.')
    transitions = validate_transitions(transitions)
    if not np.isfinite(background_cost) or background_cost < 0:
        raise ValueError('Background pairwise cost must be finite and non-negative.')
    table = np.zeros((6, 6), dtype=float)
    table[1:, 1:] = HANDWRITTEN
    table[0, 1:] = table[1:, 0] = background_cost
    offsets = forward_offsets()
    return GridProblem(
        # A pixel's original HoVer-Net class chooses its six-value cost vector.
        # Costs always refer to that original class, not the last ICM update.
        unary=transitions[initial],
        offsets=offsets,
        weights=distance_weights(offsets, 1) * bool(pairwise),
        table=table,
        active=np.ones(initial.shape, dtype=bool),
    )


def probability_unary(nucleus_probability, type_probability=None):
    """Relative -log probability costs for original TP or combined NP/TP heads.

    With one argument, consume original six-channel TP probabilities directly.
    With two, type probabilities are conditional on nucleus membership (1..5).
    Subtracting each pixel's minimum changes only the energy's constant offset.
    """
    if type_probability is None:
        values = np.asarray(nucleus_probability, dtype=float)
        if values.ndim != 3 or values.shape[0] != 6:
            raise ValueError('Original TP probabilities must have shape (6,H,W).')
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Original TP probabilities must be finite and in [0,1].')
        if not np.allclose(values.sum(axis=0), 1, rtol=0, atol=1e-5):
            raise ValueError('Original TP probabilities must sum to one at each pixel.')
        unary = -np.log(np.clip(values, 1e-8, 1)).transpose(1, 2, 0)
        return unary - unary.min(axis=-1, keepdims=True)
    p = np.asarray(nucleus_probability, dtype=float)
    types = np.asarray(type_probability, dtype=float)
    if p.ndim != 2 or types.shape != (5, *p.shape):
        raise ValueError('Expected P(nucleus) (H,W) and conditional type probabilities (5,H,W).')
    if not np.isfinite(p).all() or not np.isfinite(types).all():
        raise ValueError('Probability maps must be finite.')
    if ((p < 0) | (p > 1)).any() or ((types < 0) | (types > 1)).any():
        raise ValueError('Probabilities must be in [0,1].')
    if not np.allclose(types.sum(axis=0), 1, rtol=0, atol=1e-5):
        raise ValueError('Conditional type probabilities must sum to one at every pixel.')
    joint = np.concatenate([(1-p)[None], p[None]*types], axis=0)
    unary = -np.log(np.clip(joint, 1e-8, 1)).transpose(1, 2, 0)
    return unary - unary.min(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--unary-mode', choices=['probabilities', 'transitions'], default='probabilities',
                        help='Default: relative costs from actual probability maps. Transitions reproduces the old experiment.')
    parser.add_argument('--probabilities', type=Path,
                        help='NPZ with nucleus_probability (H,W) and type_probability (5,H,W); defaults to source/hovernet_probabilities.npz.')
    parser.add_argument('--probability-head', choices=['combined_np_tp', 'original_tp'], default='combined_np_tp',
                        help='original_tp consumes type_probability_all (6,H,W), without NP fusion.')
    parser.add_argument('--add-type-transition-prior', action='store_true',
                        help='Add calibrated type-to-type penalties to probability costs; no fixed background penalty is added.')
    parser.add_argument('--initialization', choices=['probability_argmax','hovernet'], default='hovernet',
                        help='Default retains the better-tested HoVer-Net initialization. probability_argmax tests pixel evidence alone; both use full per-pixel unaries.')
    parser.add_argument('--transition-costs', default=DEFAULT_TRANSITIONS, type=Path,
                        help='JSON 6x6 array: rows original class, columns chosen class (0..5).')
    parser.add_argument('--background-pairwise-cost', default=0.25, type=float,
                        help='Experimental background/nucleus edge cost before dividing by distance.')
    parser.add_argument('--pairwise-strength', default=1.0, type=float,
                        help='Multiply all neighbor costs by this nonnegative strength; zero leaves unary-only inference.')
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--check-unary-only', action='store_true',
                        help='Also run an independent no-pairwise sanity check; not a pipeline stage.')
    args = parser.parse_args()
    if args.max_rounds < 1:
        parser.error('--max-rounds must be positive')
    if not np.isfinite(args.pairwise_strength) or args.pairwise_strength < 0:
        parser.error('--pairwise-strength must be finite and non-negative')
    source = args.source.resolve()
    output = args.output.resolve()
    # Do not overwrite historical output or accidentally re-use a previous run.
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be a new or empty directory')
    initial = np.load(source / 'hovernet_labels_164.npy', allow_pickle=False)
    transitions = None
    if args.unary_mode == 'transitions' or args.add_type_transition_prior:
        transitions = validate_transitions(json.loads(args.transition_costs.read_text()))
    problem = make_problem(initial, np.zeros((6, 6)), args.background_pairwise_cost)
    probabilities = None
    probability_path = None
    if args.unary_mode == 'probabilities':
        probability_path = args.probabilities or source / 'hovernet_probabilities.npz'
        if not probability_path.is_file():
            parser.error(f'Missing probability maps: {probability_path}. Run export_hovernet_probabilities.py; the classification CSV is insufficient.')
        with np.load(probability_path, allow_pickle=False) as data:
            probabilities = ((data['type_probability_all'],) if args.probability_head == 'original_tp'
                             else (data['nucleus_probability'], data['type_probability']))
        if probability_unary(*probabilities).shape[:2] != initial.shape:
            parser.error('Probability maps must match the prepared prediction region.')
    elif args.probabilities or args.add_type_transition_prior:
        parser.error('Probability options require --unary-mode probabilities.')
    optimization_initial = (probability_unary(*probabilities).argmin(axis=-1)
                            if probabilities is not None and args.initialization == 'probability_argmax' else initial)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'source': str(source), 'ground_truth_used': False,
        'mode': 'all_pixels_six_classes', 'label_dictionary': dict(enumerate(LABELS)),
        'domain': list(range(6)), 'variable_count': int(initial.size),
        'unary_mode': args.unary_mode,
        'probability_head': args.probability_head,
        'initialization': args.initialization if probabilities is not None else 'hovernet',
        'clustering_used': False, 'nucleus_averaging_used': False,
        'probability_source': str(probability_path.resolve()) if probability_path else None,
        'add_type_transition_prior': args.add_type_transition_prior,
        'unary_transition_costs': (transitions.tolist() if probabilities is None else None),
        'type_transition_prior': (transitions[1:, 1:].tolist() if args.add_type_transition_prior else None),
        'unary_definition': ('relative -log P(background or nucleus type)' if probabilities is not None
                             else 'cost[original_hovernet_class, candidate_class]'),
        'transition_costs_source': str(args.transition_costs.resolve()) if transitions is not None else None,
        'background_pairwise_cost': args.background_pairwise_cost,
        'pairwise_strength': args.pairwise_strength,
        'pairwise_table': problem.table.tolist(),
        'solver': 'BlockCoordinateDescent(ICMSolver)', 'max_rounds': args.max_rounds,
        'runs': [],
    }
    settings = [('unary_only', False)] if args.check_unary_only else []
    settings.append(('refined', True))
    for name, enabled in settings:
        result, run = solve_unified(optimization_initial, transitions, args.background_pairwise_cost,
                                    args.max_rounds, pairwise=enabled, probabilities=probabilities,
                                    add_transition_prior=args.add_type_transition_prior,
                                    pairwise_strength=args.pairwise_strength)
        run['setting'] = name
        np.save(output / f'{name}_labels.npy', result.labels)
        if name == 'refined':
            np.save(output / 'active_mask.npy', result.labels > 0)
        report['runs'].append(run)
        print(json.dumps(run))
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.save(output / 'optimization_initial_labels.npy', optimization_initial)


def solve_unified(initial, transitions=None, background_cost=0.25, max_rounds=20, pairwise=True,
                  *, probabilities=None, add_transition_prior=False, pairwise_strength=1.0):
    """One joint labeling: presence and type are chosen by the same six-label solver."""
    if not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or max_rounds < 1:
        raise ValueError('max_rounds must be a positive integer')
    if not np.isfinite(pairwise_strength) or pairwise_strength < 0:
        raise ValueError('pairwise_strength must be finite and non-negative')
    if probabilities is None:
        if add_transition_prior:
            raise ValueError('A type-transition prior requires probability evidence.')
        problem = make_problem(initial, transitions, background_cost, pairwise)
    else:
        unary = probability_unary(*probabilities)
        if unary.shape[:2] != initial.shape:
            raise ValueError('Probability maps must match the initial labels.')
        if add_transition_prior:
            prior = validate_transitions(transitions).copy()
            prior[0, :] = prior[:, 0] = 0
            unary += prior[initial]
        problem = replace(make_problem(initial, np.zeros((6, 6)), background_cost, pairwise), unary=unary)
    problem = replace(problem, weights=problem.weights * pairwise_strength)
    layout = (BlockLayout.cross(initial.shape, block=60) if min(initial.shape) >= 123
              else BlockLayout.whole(initial.shape))
    result = BlockCoordinateDescent(ICMSolver(), layout=layout, max_rounds=max_rounds).solve(problem, initial)
    costs = local_costs(problem, result.labels)
    chosen = np.take_along_axis(costs, result.labels[..., None], -1)[..., 0]
    return result, {
        'initial_energy': energy(problem, initial), 'returned_energy': result.energy,
        'changed_pixels': int((result.labels != initial).sum()),
        'background_to_nucleus': int(((initial == 0) & (result.labels > 0)).sum()),
        'nucleus_to_background': int(((initial > 0) & (result.labels == 0)).sum()),
        'local_minimum': bool((costs.min(-1) >= chosen - 1e-9).all()),
        'runtime_seconds': result.seconds,
        'pairwise_strength': float(pairwise_strength),
    }


if __name__ == '__main__':
    main()
