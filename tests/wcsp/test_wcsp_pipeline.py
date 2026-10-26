import numpy as np
import pytest

from wcsp.classifiers.base import Prediction
from wcsp.classifiers.cache import ProbStore
from wcsp.classifiers.calibration import fit_temperature
from wcsp.config import ProblemConfig, apply_overrides
from wcsp.data import select
from wcsp.evaluate.compare import contrast_bootstrap, paired_bootstrap
from wcsp.evaluate.elicit import counts_by_type, fill_background, pmi_table
from wcsp.evaluate.rules import pair_counts, rule_report
from wcsp.labels import DEAD, HEALTHY, IGNORE, NEOPLASTIC, NUM_LABELS
from wcsp.model import build_problem, calibrate
from wcsp.model.tables import OTHER_NEUTRAL, load_table


def test_select_is_stable_across_shard_counts():
    whole = select(100, limit=20)
    assert len(whole) == 20 and whole[0] == 0 and whole[-1] == 99
    sharded = sorted(select(100, (0, 3), 20) + select(100, (1, 3), 20) + select(100, (2, 3), 20))
    assert sharded == whole
    with pytest.raises(ValueError):
        select(10, (3, 3))


def test_overrides_parse_yaml_values():
    values = apply_overrides({"problem": {"lam": 1}}, ["problem.lam=2.5", "solver.inner.time_limit=5"])
    assert values["problem"]["lam"] == 2.5
    assert values["solver"]["inner"]["time_limit"] == 5
    assert ProblemConfig.from_dict(values["problem"]).lam == 2.5


def test_prob_store_roundtrip(tmp_path):
    store = ProbStore(tmp_path, "bcss", "fake", "dev")
    probs = np.random.default_rng(0).dirichlet(np.ones(NUM_LABELS), size=(3, 4, 4)).transpose(0, 3, 1, 2)
    store.write(0, 2, ["a", "b"], [Prediction(probs[0]), Prediction(probs[1])])
    store.write(1, 2, ["c"], [Prediction(probs[2])])
    fresh = ProbStore(tmp_path, "bcss", "fake", "dev")
    assert sorted(fresh.ids()) == ["a", "b", "c"]
    assert np.allclose(fresh.load("c").probs, probs[2], atol=1e-3)
    store.save_temperature(1.7)
    assert fresh.temperature() == 1.7


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    target = rng.integers(0, NUM_LABELS, size=(1, 4000))
    logits = rng.normal(size=(NUM_LABELS, 4000))
    logits[target[0], np.arange(4000)] += 1.0
    honest = np.exp(logits) / np.exp(logits).sum(0)
    overconfident = calibrate(honest, temperature=1 / 3)        # logits x3
    best, _ = fit_temperature([(overconfident[:, None, :], target)])
    assert 2.0 < best < 4.5


def test_rule_counts():
    labels = np.full((4, 6), HEALTHY)
    labels[:, :2] = DEAD
    counts = pair_counts(labels, offsets=np.array([[0, 1]]))
    assert counts[DEAD, HEALTHY] == 4          # one dead-healthy contact per row
    report = rule_report(counts)
    assert report["necrosis_isolation"] == pytest.approx(4 / 20 * 1e4)
    labels[0, 0] = IGNORE
    assert pair_counts(labels, offsets=np.array([[0, 1]]))[DEAD, DEAD] == 3


def test_elicited_table_penalizes_rare_pairs():
    labels = np.full((20, 20), NEOPLASTIC)
    labels[:, 10:] = HEALTHY
    labels[:5, :5] = DEAD                   # dead only ever touches neoplastic
    table = pmi_table(sum(counts_by_type([labels]).values()), match_scale=False)
    assert np.allclose(table, table.T) and np.all(np.diag(table) == 0)
    assert table[DEAD, HEALTHY] > table[DEAD, NEOPLASTIC]
    filled = fill_background(np.array([[0, IGNORE, 3]]), np.array([[True, False, True]]))
    assert filled[0, 1] in (0, 3)


def test_other_neutral_table():
    table = load_table("other_neutral")
    assert np.all(table[4, :4] == 1) and np.array_equal(table[:4, :4], OTHER_NEUTRAL[:4, :4])


def test_paired_bootstrap_detects_a_clear_gain():
    rng = np.random.default_rng(0)
    base = rng.integers(50, 100, size=(30, NUM_LABELS, NUM_LABELS))
    better = base.copy()
    better[:, np.arange(NUM_LABELS), np.arange(NUM_LABELS)] += 200
    result = paired_bootstrap(base, better, np.repeat(np.arange(10), 3), n_boot=200)
    assert result["delta_mIoU"] > 0 and result["ci_low"] > 0 and result["groups"] == 10


def test_contrast_bootstrap_matches_paired_bootstrap():
    rng = np.random.default_rng(1)
    a = rng.integers(50, 100, size=(30, NUM_LABELS, NUM_LABELS))
    b = a + rng.integers(0, 30, size=a.shape)
    groups = np.repeat(np.arange(10), 3)
    paired = paired_bootstrap(a, b, groups, n_boot=200)
    contrast = contrast_bootstrap({"a": a, "b": b}, groups, {"b": 1, "a": -1}, n_boot=200)
    assert contrast == pytest.approx(paired)


def test_shift_stitch_recovers_the_stride_one_output():
    torch = pytest.importorskip("torch")
    from wcsp.classifiers.tia_unet import shift_stitch

    torch.manual_seed(0)
    weight = torch.randn(2, 3, 3, 3)
    x = torch.randn(1, 3, 16, 16)
    stitched = shift_stitch(lambda v: torch.nn.functional.conv2d(v, weight, stride=2, padding=1), x)
    dense = torch.nn.functional.conv2d(x, weight, stride=1, padding=1)
    # Shifted copies move the zero padding by a pixel and reflect their last
    # row / column, so only the interior must agree.
    inner = (..., slice(2, -2), slice(2, -2))
    assert torch.allclose(stitched[inner], dense[inner], atol=1e-5)


def test_toulbar2_matches_brute_force():
    pytest.importorskip("pytoulbar2")
    import itertools

    from wcsp.solve import energy
    from wcsp.solve.toulbar2 import Toulbar2Solver

    rng = np.random.default_rng(3)
    logits = rng.normal(size=(NUM_LABELS, 2, 4))
    probs = np.exp(logits) / np.exp(logits).sum(0)
    problem = build_problem(probs, ProblemConfig(lam=8.0))
    best = min(energy(problem, np.array(a).reshape(2, 4))
               for a in itertools.product(range(NUM_LABELS), repeat=8))
    result = Toulbar2Solver(time_limit=10).solve(problem, probs.argmax(0))
    assert result.energy == pytest.approx(best, abs=1e-2)


def _tiny_fullres_unet():
    torch = pytest.importorskip("torch")
    unet = pytest.importorskip("tiatoolbox.models.architecture.unet")
    from wcsp.classifiers.fullres_unet import FullResUNet

    torch.manual_seed(0)
    base = unet.UNetModel(num_input_channels=3, num_output_channels=NUM_LABELS, encoder="unet",
                          encoder_levels=[8, 16, 32], decoder_block=[3]).eval()
    return torch, base, FullResUNet(base, width=4)


def test_untrained_fullres_head_reproduces_the_base():
    torch, base, model = _tiny_fullres_unet()
    x = torch.rand(2, 3, 64, 64) * 255
    with torch.no_grad():
        expected = torch.nn.functional.interpolate(base(x), size=(64, 64), mode="bilinear",
                                                   align_corners=False)
        assert torch.allclose(model.eval()(x), expected, atol=1e-5)


def test_frozen_base_is_not_trained():
    torch, base, model = _tiny_fullres_unet()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.train()
    for _ in range(2):
        loss = model(torch.rand(2, 3, 64, 64) * 255).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())   # incl. batch-norm stats
    assert model.clf.weight.abs().sum() > 0
    assert not any(k.startswith("base.") for k in model.head_state_dict())
