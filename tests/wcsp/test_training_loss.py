import numpy as np
import pytest

torch = pytest.importorskip("torch")
from wcsp.classifiers.training_loss import expected_pairwise_cost
from wcsp.labels import BCSS_TO_WCSP
from wcsp.model.problem import build_problem
from wcsp.solve.energy import energy
from wcsp.config import ProblemConfig


def test_expected_cost_matches_discrete_wcsp_pairwise_energy():
    rng = np.random.default_rng(7)
    labels = rng.integers(5, size=(6, 7))
    probs = torch.nn.functional.one_hot(torch.tensor(labels), 5).permute(2, 0, 1).double()[None]
    target = torch.tensor(labels)[None]
    expected = expected_pairwise_cost(probs, target).item()
    wcsp_labels = BCSS_TO_WCSP[labels]
    pr = build_problem(np.full((5, 6, 7), .2), ProblemConfig(alpha=0))
    assert expected == pytest.approx(energy(pr, wcsp_labels)/labels.size)


def test_loss_has_finite_gradients_and_ignores_unlabeled_pairs():
    logits = torch.randn(2, 5, 5, 6, requires_grad=True)
    probs = logits.softmax(1)
    ignored = torch.full((2, 5, 6), 255)
    zero = expected_pairwise_cost(probs, ignored)
    assert zero.item() == 0
    zero.backward(retain_graph=True)
    assert logits.grad.abs().sum() == 0
    logits.grad = None
    loss = expected_pairwise_cost(probs, torch.zeros_like(ignored))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_fullres_prediction_uses_each_samples_window():
    from wcsp.classifiers.tia_unet import FullResTiaUNet
    from wcsp.data.base import Sample
    # Exercise the inference window contract without downloading weights.
    classifier = object.__new__(FullResTiaUNet)
    classifier.torch = torch
    classifier.device = "cpu"
    classifier.model = torch.nn.Conv2d(3, 5, 1)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    full = Sample("full", image, None)
    center = Sample("center", image, None, window=(slice(128, 384), slice(128, 384)))
    predictions = classifier.predict([full, center])
    assert predictions[0].probs.shape == (5, 512, 512)
    assert predictions[1].probs.shape == (5, 256, 256)
    np.testing.assert_allclose(predictions[0].probs.sum(0), 1, atol=1e-6)
