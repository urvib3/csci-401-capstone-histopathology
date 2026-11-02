"""Checks for the PanNuke-only six-label experiment."""

import itertools

import numpy as np
import pytest

from experiments.pannuke_d5.evaluate_fullgrid import evaluate
from experiments.pannuke_d5.run_fullgrid import (
    DEFAULT_TRANSITIONS, make_problem, probability_unary, solve_unified, validate_transitions,
)
from wcsp.model.tables import HANDWRITTEN
from wcsp.solve.energy import energy
from wcsp.solve.icm import ICMSolver


def costs():
    matrix = np.full((6, 6), 100.0)
    np.fill_diagonal(matrix, 0)
    return matrix


def test_original_tp_unary_retains_all_six_channels_and_probability_ratios():
    probabilities = np.array([.1, .55, .2, .1, .02, .03])[:, None, None]
    unary = probability_unary(probabilities)
    assert unary.shape == (1, 1, 6)
    assert unary.argmin(-1).item() == 1
    assert unary[0, 0, 0] == pytest.approx(np.log(.55/.1))
    result, _ = solve_unified(np.zeros((1, 1), dtype=int), pairwise=False,
                              probabilities=(probabilities,))
    assert result.labels.item() == 1


def test_original_tp_unary_rejects_unnormalized_values():
    with pytest.raises(ValueError, match='sum to one'):
        probability_unary(np.full((6, 2, 2), .5))


def test_zero_pairwise_strength_is_probability_argmax_not_saved_baseline():
    probabilities = np.full((6, 2, 2), .02)
    probabilities[2] = .9
    initial = np.ones((2, 2), dtype=int)
    result, report = solve_unified(initial, probabilities=(probabilities,), pairwise_strength=0)
    np.testing.assert_array_equal(result.labels, np.full((2, 2), 2))
    assert report['returned_energy'] <= report['initial_energy']


@pytest.mark.parametrize('strength', [-1, float('nan'), float('inf')])
def test_pairwise_strength_must_be_finite_and_nonnegative(strength):
    with pytest.raises(ValueError, match='pairwise_strength'):
        solve_unified(np.zeros((1, 1), dtype=int), costs(), pairwise_strength=strength)


def test_directed_costs_use_the_original_class():
    matrix = costs()
    matrix[3, 2], matrix[2, 3] = .25, 7
    initial = np.array([[3, 2, 0]])
    problem = make_problem(initial, matrix, 1, pairwise=False)
    assert problem.unary[0, 0, 2] == .25
    assert problem.unary[0, 1, 3] == 7
    assert problem.num_variables == 3
    assert problem.num_labels == 6
    np.testing.assert_array_equal(problem.table[1:, 1:], HANDWRITTEN)
    np.testing.assert_array_equal(ICMSolver().solve(problem, initial).labels, initial)


@pytest.mark.parametrize('initial,source,target', [([[0, 2, 2]], 0, 2), ([[2, 0, 0]], 2, 0)])
def test_background_can_be_added_or_removed(initial, source, target):
    initial = np.array(initial)
    matrix = costs()
    matrix[source, target] = .1
    problem = make_problem(initial, matrix, 4)
    result = ICMSolver().solve(problem, initial)
    assert result.labels[0, 0] == target
    assert result.energy < energy(problem, initial)
    # Small independent exact enumeration checks the solver's energy and result.
    best = min(energy(problem, np.array(x).reshape(initial.shape))
               for x in itertools.product(range(6), repeat=initial.size))
    assert result.energy == pytest.approx(best)


def test_background_corrections_and_errors_count_in_evaluation():
    initial = np.array([[0, 2, 0, 2]])
    refined = np.array([[2, 0, 2, 0]])
    gt = np.array([[2, 0, 0, 2]])
    report = evaluate(initial, refined, gt)
    assert report['wrong_to_correct'] == 2
    assert report['correct_to_wrong'] == 2
    assert report['changed_pixels'] == 4
    assert report['hovernet']['missed_nucleus_pixels'] == 1
    assert report['wcsp']['false_positive_nucleus_pixels'] == 1


@pytest.mark.parametrize('bad', [np.ones((5, 5)), np.ones((6, 6)),
                                      np.eye(6) * np.nan, -np.ones((6, 6))])
def test_invalid_unary_matrices_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_transitions(bad)


def test_same_optimization_changes_background_and_type():
    initial = np.array([[0, 2, 3]])
    matrix = costs()
    matrix[0, 3] = .1
    matrix[2, 3] = .1
    result, report = solve_unified(initial, matrix, background_cost=4)
    np.testing.assert_array_equal(result.labels, [[3, 3, 3]])
    assert report['background_to_nucleus'] == 1
    assert report['changed_pixels'] == 2
    assert report['local_minimum']
    assert report['returned_energy'] < report['initial_energy']


def test_default_table_combines_background_and_calibrated_type_costs():
    import json
    matrix = validate_transitions(json.loads(DEFAULT_TRANSITIONS.read_text()))
    np.testing.assert_array_equal(matrix[0, 1:], 2)
    np.testing.assert_array_equal(matrix[1:, 0], 2)
    assert matrix[1, 3] != matrix[3, 1]
    assert np.isclose(matrix[1:, 1:][~np.eye(5, dtype=bool)].mean(), 1)


def test_joint_unary_only_keeps_original_labels_without_an_intermediate_mask():
    import json
    initial = np.array([[0, 1, 2], [3, 4, 5]])
    matrix = json.loads(DEFAULT_TRANSITIONS.read_text())
    result, report = solve_unified(initial, matrix, pairwise=False)
    np.testing.assert_array_equal(result.labels, initial)
    assert report['changed_pixels'] == 0


def test_probability_costs_are_relative_and_preserve_joint_odds():
    p = np.array([[.1, .9]])
    types = np.zeros((5, 1, 2)) + .05
    types[0] = .8
    unary = probability_unary(p, types)
    # U(type1)-U(background) = log(P(background)/(P(nucleus)*P(type1|nucleus))).
    np.testing.assert_allclose(unary[..., 1]-unary[..., 0], np.log((1-p)/(p*.8)))
    assert unary[0, 0, 0] == 0
    assert unary[0, 1, 1] == 0
    assert unary[0, 0, 1] > unary[0, 1, 1]


def test_probability_unary_only_chooses_joint_argmax_not_saved_labels():
    initial = np.array([[3, 0]])
    p = np.array([[.01, .99]])
    types = np.zeros((5, 1, 2)) + .025
    types[1] = .9
    result, _ = solve_unified(initial, probabilities=(p, types), pairwise=False)
    np.testing.assert_array_equal(result.labels, [[0, 2]])


def test_probability_extremes_are_finite():
    unary = probability_unary(np.array([[0., 1.]]), np.ones((5, 1, 2))/5)
    assert np.isfinite(unary).all()
    assert (unary >= 0).all()


@pytest.mark.parametrize('p,types', [
    (np.array([[1.1]]), np.ones((5, 1, 1))/5),
    (np.array([[np.nan]]), np.ones((5, 1, 1))/5),
    (np.array([[.5]]), np.ones((5, 1, 1))),
    (np.array([[.5]]), np.ones((6, 1, 1))/6),
])
def test_invalid_probability_maps_are_rejected(p, types):
    with pytest.raises(ValueError):
        probability_unary(p, types)


def test_optional_type_prior_never_reintroduces_fixed_background_penalty():
    prior = costs()  # Background transitions cost 100, but are ignored in probability mode.
    initial = np.array([[0]])
    p = np.array([[.99]])
    types = np.array([.9, .025, .025, .025, .025]).reshape(5, 1, 1)
    result, _ = solve_unified(initial, prior, probabilities=(p, types),
                              add_transition_prior=True, pairwise=False)
    np.testing.assert_array_equal(result.labels, [[1]])


def test_cli_probability_mode_records_evidence_and_has_no_fixed_background_unary(tmp_path, monkeypatch):
    import json
    from experiments.pannuke_d5.run_fullgrid import main
    source = tmp_path/'source'
    source.mkdir()
    initial = np.array([[0, 0]])
    np.save(source/'hovernet_labels_164.npy', initial)
    p = np.array([[.001, .999]])
    types = np.array([.9, .025, .025, .025, .025]).reshape(5, 1, 1).repeat(2, axis=2)
    np.savez(source/'hovernet_probabilities.npz', nucleus_probability=p, type_probability=types)
    out = tmp_path/'output'
    monkeypatch.setattr('sys.argv', ['run_fullgrid.py','--source',str(source),'--output',str(out),
                                   '--background-pairwise-cost','0','--initialization','probability_argmax'])
    main()
    report = json.loads((out/'results.json').read_text())
    assert report['unary_mode'] == 'probabilities'
    assert report['unary_transition_costs'] is None
    assert report['type_transition_prior'] is None
    assert report['ground_truth_used'] is False
    assert report['initialization'] == 'probability_argmax'
    assert report['clustering_used'] is False
    assert report['nucleus_averaging_used'] is False
    assert len(report['runs']) == 1
    np.testing.assert_array_equal(np.load(out/'optimization_initial_labels.npy'), [[0, 1]])
    np.testing.assert_array_equal(np.load(out/'refined_labels.npy'), [[0, 1]])


def test_pixel_evidence_is_not_averaged_for_pixels_with_same_original_type():
    initial = np.array([[2, 2]])
    p = np.array([[.99, .99]])
    types = np.full((5, 1, 2), .025)
    types[1, 0, 0] = .9
    types[2, 0, 1] = .9
    unary = probability_unary(p, types)
    assert unary.shape == (1, 2, 6)
    assert not np.array_equal(unary[0, 0], unary[0, 1])
    pixel_initial = unary.argmin(-1)
    result, _ = solve_unified(pixel_initial, probabilities=(p, types), pairwise=False)
    np.testing.assert_array_equal(result.labels, [[2, 3]])


def test_cli_missing_probabilities_fails_without_fabricating_them(tmp_path, monkeypatch):
    from experiments.pannuke_d5.run_fullgrid import main
    source = tmp_path/'source'
    source.mkdir()
    np.save(source/'hovernet_labels_164.npy', np.array([[0]]))
    out = tmp_path/'output'
    monkeypatch.setattr('sys.argv', ['run_fullgrid.py','--source',str(source),'--output',str(out)])
    with pytest.raises(SystemExit):
        main()
    assert not out.exists()
