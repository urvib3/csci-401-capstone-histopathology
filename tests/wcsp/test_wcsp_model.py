import itertools

import numpy as np
import pytest

from wcsp.config import ProblemConfig
from wcsp.labels import (
    BCSS_TO_WCSP, IGNORE, LABELS, NUM_LABELS, colorize, remap_labels, reorder_probs,
)
from wcsp.model import (
    HANDWRITTEN, build_problem, contrast_term, distance_weights, forward_offsets,
    instance_term, load_table, neighbor_type, pair_slices, save_table, unary_costs,
)


def random_probs(shape, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(NUM_LABELS,) + shape)
    probs = np.exp(logits)
    return probs / probs.sum(axis=0)


def test_bcss_labels_map_to_wcsp_order():
    # BCSS coarse order: tumor, stroma, inflammatory, necrosis, other
    bcss = np.array([[0, 1, 2], [3, 4, IGNORE]], dtype=np.uint8)
    wcsp = remap_labels(bcss, BCSS_TO_WCSP)
    names = [[LABELS[v] if v != IGNORE else None for v in row] for row in wcsp]
    assert names == [["neoplastic", "connective", "inflammatory"], ["dead", "healthy", None]]


def test_reorder_probs_matches_remap_labels():
    probs = random_probs((4, 5))
    reordered = reorder_probs(probs, BCSS_TO_WCSP)
    assert np.array_equal(reordered.argmax(0), remap_labels(probs.argmax(0), BCSS_TO_WCSP))
    with pytest.raises(ValueError):
        reorder_probs(probs, [0, 0, 1, 2, 3])
    assert colorize(np.array([[0, IGNORE]])).shape == (1, 2, 3)


def test_forward_offsets_cover_the_five_neighbor_types_once():
    offsets = forward_offsets()
    assert len(offsets) == 12
    both = {tuple(o) for o in offsets} | {tuple(-o) for o in offsets}
    assert len(both) == 24
    counts = {}
    for o in both:
        counts[neighbor_type(o)] = counts.get(neighbor_type(o), 0) + 1
    assert counts == {1: 4, 2: 4, 3: 4, 4: 8, 5: 4}
    assert len(forward_offsets((1,))) == 2


def test_distance_weights():
    offsets = forward_offsets()
    weights = distance_weights(offsets, eta=2)
    for o, w in zip(offsets, weights):
        assert w == pytest.approx(1 / (o ** 2).sum())
    assert np.allclose(distance_weights(offsets, eta=0), 1)
    assert 2 * distance_weights(offsets, eta=1, normalize=True).sum() == pytest.approx(1)


def test_pair_slices_count_every_pair_once():
    shape = (7, 9)
    count = 0
    for offset in forward_offsets():
        p, q = pair_slices(shape, offset)
        rows = np.indices(shape)
        assert np.array_equal(rows[0][q] - rows[0][p], np.full(rows[0][p].shape, offset[0]))
        assert np.array_equal(rows[1][q] - rows[1][p], np.full(rows[1][p].shape, offset[1]))
        count += rows[0][p].size
    pixels = list(itertools.product(range(shape[0]), range(shape[1])))
    brute = sum(
        1 for a, b in itertools.combinations(pixels, 2)
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= 8
    )
    assert count == brute


def test_handwritten_table_and_csv_roundtrip(tmp_path):
    table = load_table("handwritten")
    assert np.allclose(table, table.T) and np.all(np.diag(table) == 0)
    # Stromal boundary: tumor-healthy (9) costs more than going through stroma (1 + 1).
    assert table[0, 4] > table[0, 2] + table[2, 4]
    path = tmp_path / "table.csv"
    save_table(HANDWRITTEN, path)
    assert np.array_equal(load_table(str(path)), HANDWRITTEN)
    with pytest.raises(ValueError):
        load_table(np.ones((5, 5)) - np.eye(5) + np.triu(np.ones((5, 5)), 1))


def test_unary_costs():
    probs = random_probs((6, 6))
    costs = unary_costs(probs)
    assert costs.shape == (6, 6, NUM_LABELS)
    assert np.array_equal(costs.argmin(-1), probs.argmax(0))
    assert np.allclose(costs.min(-1), 0)
    # Temperature softens costs but keeps the argmax.
    soft = unary_costs(probs, temperature=3.0)
    assert np.array_equal(soft.argmin(-1), probs.argmax(0))
    assert soft.max() < costs.max()
    informative = np.ones((6, 6), dtype=bool)
    informative[0, 0] = False
    assert np.all(unary_costs(probs, informative=informative)[0, 0] == 0)


def test_contrast_and_instance_terms():
    offsets = forward_offsets()
    features = np.zeros((4, 4, 3))
    features[:, 2:] = 1.0  # a vertical colour edge between columns 1 and 2
    term = contrast_term(features, offsets, beta=2.0, sigma=0.1)
    right = [i for i, o in enumerate(offsets) if tuple(o) == (0, 1)][0]
    assert term[right, 0, 0] == pytest.approx(2.0)   # same colour
    assert term[right, 0, 1] < 1e-6                  # across the edge
    assert term[right, 0, 3] == 0                    # q outside the image

    instances = np.array([[1, 1, 0, 2]])
    term = instance_term(instances, offsets, gamma=5.0)
    assert term[right, 0].tolist() == [5.0, 0.0, 0.0, 0.0]


def test_build_problem_from_config():
    probs = random_probs((8, 8))
    image = np.random.default_rng(1).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    config = ProblemConfig.from_dict({"alpha": 2.0, "eta": 2, "types": [1, 2], "contrast_beta": 1.0})
    problem = build_problem(probs, config, image=image)
    assert problem.unary.shape == (8, 8, NUM_LABELS)
    assert len(problem.offsets) == 4 and problem.change.shape == (4, 8, 8)
    assert problem.num_variables == 64
    with pytest.raises(ValueError):
        build_problem(probs, ProblemConfig(contrast_beta=1.0))
    with pytest.raises(ValueError):
        ProblemConfig.from_dict({"alpah": 1})
