import numpy as np
import pytest

from bcss.classes import IGNORE_INDEX, coarse_lut, full_lut
from bcss.grid import CellGrid, to_integer_costs, unary_costs_from_probabilities
from bcss.metrics import confusion_matrix, iou_per_class, mean_iou, pixel_accuracy
from bcss.stats import adjacency_counts, class_counts, negative_log_costs


def test_coarse_lut_maps_codes_and_ignores_dont_care():
    lut, names = coarse_lut()
    assert names == ["tumor", "stroma", "inflammatory", "necrosis", "other"]
    raw = np.array([0, 1, 2, 3, 4, 7, 10, 20, 21])
    assert lut[raw].tolist() == [IGNORE_INDEX, 0, 1, 2, 3, IGNORE_INDEX, 2, 0, 4]


def test_full_lut_has_twenty_classes():
    lut, names = full_lut()
    assert len(names) == 20
    assert lut[0] == lut[7] == IGNORE_INDEX
    assert lut[1] == 0 and lut[21] == 19


def test_cell_grid_edges():
    grid = CellGrid(4, 6, 2)  # 2 x 3 lattice
    assert (grid.rows, grid.cols, grid.num_cells) == (2, 3, 6)
    edges4 = {tuple(e) for e in grid.edges(4)}
    assert edges4 == {(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)}
    edges8 = {tuple(e) for e in grid.edges(8)}
    assert edges8 - edges4 == {(0, 4), (1, 5), (1, 3), (2, 4)}


def test_cell_grid_rejects_bad_size():
    with pytest.raises(ValueError):
        CellGrid(10, 10, 3)


def test_majority_labels_and_upsample():
    mask = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [2, 2, IGNORE_INDEX, IGNORE_INDEX],
        [2, 1, IGNORE_INDEX, IGNORE_INDEX],
    ])
    grid = CellGrid(4, 4, 2)
    hist = grid.label_histogram(mask, 3)
    assert hist[0, 0].tolist() == [3, 1, 0]
    assert hist[1, 1].tolist() == [0, 0, 0]
    labels = grid.majority_labels(mask, 3)
    assert labels.tolist() == [[0, 1], [2, IGNORE_INDEX]]
    assert grid.upsample(labels).shape == (4, 4)
    assert grid.upsample(labels)[3, 1] == 2


def test_pool_probabilities_and_unary_costs():
    probs = np.zeros((2, 4, 4))
    probs[0, :2, :2] = 1.0
    probs[1] = 1.0 - probs[0]
    pooled = CellGrid(4, 4, 2).pool_probabilities(probs)
    assert pooled.shape == (2, 2, 2)
    assert pooled[0, 0].tolist() == [1.0, 0.0]
    costs = unary_costs_from_probabilities(pooled)
    assert costs.shape == (4, 2)
    assert costs[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert costs[0, 1] > costs[0, 0]


def test_to_integer_costs_preserves_binary_table_structure():
    table = np.array([[1.0, 3.0], [2.0, 1.5]])
    assert to_integer_costs(table, scale=10).tolist() == [[0, 20], [10, 5]]
    unary = np.array([[1.0, 2.0], [5.0, 4.0]])
    assert to_integer_costs(unary, scale=1, axis=-1).tolist() == [[0, 1], [1, 0]]


def test_adjacency_counts():
    grid = np.array([[0, 0, 1], [0, 1, IGNORE_INDEX]])
    counts = adjacency_counts(grid, 2)
    # Horizontal: (0,0), (0,1), (0,1). Vertical: (0,0), (0,1). Ignored pair dropped.
    assert counts.tolist() == [[2, 3], [3, 0]]
    costs = negative_log_costs(counts)
    assert costs[0, 1] < costs[1, 1]


def test_class_counts_skip_ignore():
    masks = [np.array([[0, 1], [IGNORE_INDEX, 1]]), np.array([[2, 2]])]
    assert class_counts(masks, 3).tolist() == [1, 2, 2]


def test_metrics():
    target = np.array([0, 0, 1, 1, IGNORE_INDEX])
    pred = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(pred, target, 2)
    assert cm.tolist() == [[1, 1], [0, 2]]
    assert pixel_accuracy(cm) == pytest.approx(0.75)
    assert iou_per_class(cm).tolist() == pytest.approx([0.5, 2 / 3])
    assert mean_iou(cm) == pytest.approx((0.5 + 2 / 3) / 2)
