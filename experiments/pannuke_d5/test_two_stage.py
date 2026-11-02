import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pannuke_d5.run_two_stage import (
    binary_problem, graph_energy, nucleus_edges, nucleus_unaries,
    reconstruct_instances, run_pipeline, solve_types,
)
from experiments.pannuke_d5.evaluate_fullgrid import binary_scores
from wcsp.solve.icm import ICMSolver


def config():
    return json.loads(Path(__file__).with_name('two_stage_config.json').read_text())


def test_stage1_directed_transition_costs_and_all_pixels():
    problem = binary_problem(np.array([[False, True]]), [[0, 7], [2, 0]], 0)
    np.testing.assert_array_equal(problem.unary, [[[0, 7], [2, 0]]])
    assert problem.num_variables == 2
    assert problem.num_labels == 2


def test_stage1_probability_evidence_can_fix_both_directions():
    initial = np.array([[0, 1]])
    problem = binary_problem(initial, [[0, 1], [1, 0]], 0, np.array([[.99, .01]]))
    result = ICMSolver().solve(problem, initial)
    np.testing.assert_array_equal(result.labels, [[1, 0]])


def test_reconstruction_preserves_touching_nuclei_and_assigns_new_component():
    active = np.array([[1, 1, 1, 1, 0, 1]], dtype=bool)
    original = np.array([[3, 0, 0, 8, 0, 0]])
    instances = reconstruct_instances(active, original)
    assert instances[0, 0] != instances[0, 3]
    assert instances[0, 0] == instances[0, 1]
    assert instances[0, 3] == instances[0, 2]
    assert instances[0, 5] not in (0, instances[0, 0], instances[0, 3])
    np.testing.assert_array_equal(instances > 0, active)


def test_removed_pixels_split_a_surviving_marker():
    instances = reconstruct_instances(np.array([[1, 0, 1]], bool), np.array([[8, 8, 8]]))
    assert instances[0, 0] != instances[0, 2]


def test_new_nucleus_requires_type_evidence():
    instances = np.array([[1]])
    original = np.array([[0]])
    transitions = 1-np.eye(5)
    with pytest.raises(ValueError, match='no saved type evidence'):
        nucleus_unaries(instances, original, transitions)
    p = np.array([.05, .8, .05, .05, .05]).reshape(5, 1, 1)
    unary, initial = nucleus_unaries(instances, original, transitions, p)
    assert initial[0] == 1  # Inflammatory, internal class index 1.
    with pytest.raises(ValueError, match='uniform type evidence'):
        nucleus_unaries(instances, original, transitions, np.ones((5, 1, 1)) / 5)


def test_directed_type_costs_are_per_nucleus_and_use_original_classes():
    transitions = np.ones((5, 5))-np.eye(5)
    transitions[1, 2], transitions[2, 1] = 7, .25
    unary, initial = nucleus_unaries(np.array([[1, 1, 2]]), np.array([[2, 2, 3]]), transitions)
    assert unary[0, 2] == 7
    assert unary[1, 1] == .25
    np.testing.assert_array_equal(initial, [1, 2])


def test_nucleus_graph_crosses_background_and_matches_exact_small_solution():
    instances = np.array([[1, 0, 2]])
    edges = nucleus_edges(instances, 2)
    assert edges == [(0, 1, .5)]
    assert nucleus_edges(instances, 1) == []
    unary = np.array([[0, 10, 10, 10, 10], [1, 0, 10, 10, 10]], float)
    initial = np.array([0, 1])
    labels, trace, local = solve_types(unary, initial, edges, 2, 20)
    exact = min(graph_energy(np.array(x), unary, edges, 2)
                for x in itertools.product(range(5), repeat=2))
    assert trace[-1] == pytest.approx(exact)
    assert all(a >= b for a, b in zip(trace, trace[1:]))
    assert local
    np.testing.assert_array_equal(labels, [0, 0])


def test_pipeline_keeps_one_class_per_instance_and_mask_fixed_in_stage2():
    cfg = config()
    cfg['stage1_boundary_cost'] = 0
    labels = np.array([[0, 2, 2, 0, 3], [0, 2, 2, 0, 3]])
    original = np.array([[0, 9, 9, 0, 17], [0, 9, 9, 0, 17]])
    active, instances, before, after, report = run_pipeline(labels, original, cfg)
    np.testing.assert_array_equal(active, labels > 0)
    np.testing.assert_array_equal(after > 0, active)
    np.testing.assert_array_equal(before > 0, active)
    for i in np.unique(instances[active]):
        assert len(np.unique(after[instances == i])) == 1
    assert report['ground_truth_used'] is False
    assert report['stage2']['variable_count'] == 2


def test_empty_foreground_is_valid():
    labels = np.zeros((2, 3), dtype=int)
    active, instances, _, final, report = run_pipeline(labels, labels, config())
    assert not active.any() and not instances.any() and not final.any()
    assert report['stage2']['variable_count'] == 0


def test_binary_metrics_measure_boundaries_not_cell_types():
    gt = np.array([[0, 3, 2]])
    metrics = binary_scores(np.array([[0, 1, 0]], bool), gt)
    assert metrics['nucleus_iou'] == .5
    assert metrics['missed_nucleus_pixels'] == 1


@pytest.mark.parametrize('bad', [np.ones((2, 2)), np.array([[0, -1], [1, 0]]),
                              np.array([[0, np.nan], [1, 0]])])
def test_bad_transition_matrices_rejected(bad):
    with pytest.raises(ValueError):
        binary_problem(np.array([[False]]), bad, 1)


def test_probability_shape_and_normalization_are_checked():
    with pytest.raises(ValueError, match='shape'):
        binary_problem(np.zeros((2, 2)), [[0, 1], [1, 0]], 1, np.zeros((1, 2)))
    with pytest.raises(ValueError, match='sum to one'):
        nucleus_unaries(np.array([[1]]), np.array([[2]]), 1-np.eye(5), np.ones((5, 1, 1)))
