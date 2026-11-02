import numpy as np
import pytest

from experiments.pannuke_d5.fit_unary_transitions import (
    calibration_counts, choose_calibration, costs_from_counts,
)


def test_class_dependent_costs_reward_supported_corrections_and_allow_direction():
    counts = np.eye(5) * 30
    counts[0, 1], counts[1, 0], counts[0, 2] = 15, 1, 1
    costs, scale = costs_from_counts(counts)
    assert costs[0, 1] < costs[0, 2]
    assert costs[0, 1] < costs[1, 0]
    assert scale > 0
    assert np.isfinite(costs).all() and (costs >= 0).all()
    np.testing.assert_array_equal(costs.diagonal(), 0)
    assert costs[~np.eye(5, dtype=bool)].mean() == pytest.approx(1)


def test_empty_calibration_is_smoothed_not_divided_by_zero():
    costs, _ = costs_from_counts(np.zeros((5, 5)))
    np.testing.assert_allclose(costs, 1-np.eye(5))


def test_each_nucleus_has_unit_weight_and_false_detections_are_excluded():
    labels = np.array([[1, 1, 1, 1, 2, 3, 3, 3]])
    instances = np.array([[7, 7, 7, 7, 8, 9, 9, 9]])
    gt = np.array([[1, 1, 2, 2, 2, 0, 0, 3]])
    counts, used = calibration_counts(labels, instances, gt)
    assert used == 2
    assert counts.sum() == 2
    assert counts[0, 0] == counts[0, 1] == .5
    assert counts[1, 1] == 1
    assert counts[2].sum() == 0


def test_calibration_selection_is_reproducible_and_excludes_reserved_patches():
    pool = [{'image_index':str(i), 'has_hovernet_predictions':'True',
             'gt_nucleus_pixels':'10', 'gt_dead_pixels':str(i % 3),
             'tissue_type':f'Tissue{i % 4}'} for i in range(30)]
    selected = choose_calibration(pool, {0, 3, 7}, 15, 401)
    ids = [int(r['image_index']) for r in selected]
    assert len(ids) == len(set(ids)) == 15
    assert not set(ids) & {0, 3, 7}
    assert selected == choose_calibration(pool, {0, 3, 7}, 15, 401)
