"""Checks that pooled IoU includes background and false-positive classes."""
import numpy as np

from experiments.pannuke_d5.reevaluate_with_background import confusion, ious


def test_background_false_positive_and_missed_nucleus_both_reduce_iou():
    gt = np.array([[0, 0, 1, 1]])
    pred = np.array([[0, 1, 0, 1]])
    matrix = confusion(gt, pred)
    np.testing.assert_array_equal(matrix[:2, :2], [[1, 1], [1, 1]])
    np.testing.assert_allclose(ious(matrix)[:2], [1/3, 1/3])
    assert np.isnan(ious(matrix)[2:]).all()


def test_predicted_class_absent_from_ground_truth_is_penalized():
    matrix = confusion(np.array([[0, 0]]), np.array([[0, 2]]))
    assert ious(matrix)[0] == .5
    assert ious(matrix)[2] == 0
    assert np.nanmean(ious(matrix)) == .25
