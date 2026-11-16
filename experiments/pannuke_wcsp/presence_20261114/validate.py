"""Validate exact objectives on enumerable cases and check final paired results."""
import importlib.util
from itertools import product

import numpy as np

from prepare_fresh import CACHE, HERE, source


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    instance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    return instance


def main():
    run = module("global_wcsp", "run.py")
    star = module("pixel_wcsp", "raw_star.py")
    rng = np.random.default_rng(20261116)
    for n in [1, 2, 3, 4]:
        unary = rng.random((n, 5))
        expected = rng.random(5) * n
        assignments = np.array(list(product(range(5), repeat=n)))
        masks = np.bitwise_or.reduce(1 << assignments, axis=1)
        costs = unary[np.arange(n), assignments].sum(1)
        labels, energies = run.exact_set_options(unary)
        for i, mask in enumerate(run.MASKS):
            correct = costs[masks == mask].min(initial=np.inf)
            np.testing.assert_allclose(energies[i], correct)
            if np.isfinite(correct):
                assert np.bitwise_or.reduce(1 << labels[i]) == mask
        strength = .7

        def count_cost(labels):
            counts = np.bincount(labels, minlength=5)
            return strength * np.sum((counts - expected)**2 / (expected + 1))

        answer = run.exact_counts(unary, expected, strength)
        optimum = min(c + count_cost(a) for c, a in zip(costs, assignments))
        np.testing.assert_allclose(unary[np.arange(n), answer].sum() + count_cost(answer), optimum)

    source.HERE = HERE
    split = source.load(HERE / "split.json")
    ids = [r["image_index"] for r in split["test"]]
    assert len(ids) == len(set(ids)) == 100
    assert not set(ids) & set(split["excluded_ids"])
    evaluation = source.load(HERE / "evaluation.json")
    with np.load(HERE / "confusions.npz") as file:
        cms = dict(file)
    with np.load(HERE / "raw_only_confusions.npz") as file:
        np.testing.assert_array_equal(cms["raw_tp"], file["raw"])
        cms["raw_only"] = file["wcsp"]
    selected = source.load(HERE / "raw_only_frozen.json")
    for i, (rec, patch) in enumerate(zip(split["test"], star.inputs(split["test"]))):
        index = rec["image_index"]
        assert np.array_equal(star.refine(patch, 0, 256, False), patch[1])
        output = star.refine(patch, **selected)
        np.testing.assert_array_equal(star.confusion(patch[0], output), cms["raw_only"][i])
        with np.load(CACHE / f"pred_{index:04d}.npz") as file:
            mask = file["instances"]
        for name in evaluation["results"]:
            labels = np.load(CACHE / "outputs" / name / f"{index:04d}.npy")
            assert len(labels) == mask.max() and np.all((labels > 0) & (labels <= 5))
            np.testing.assert_array_equal(star.confusion(patch[0], np.r_[0, labels][mask]), cms[name][i])
    draws = np.random.default_rng(20261116).integers(100, size=(5000, 100))
    scores, sensitivity = {}, {}
    for name, cm in cms.items():
        pooled = star.iou(cm[draws].sum(1))
        scores[name] = pooled[:, 1:].mean(1)
        other = [1, 2, 3, 5]
        point = star.iou(cm.sum(0))[other].mean()
        sensitivity[name] = (float(point), pooled[:, other].mean(1))
    checks = dict(evaluation["comparisons"])
    checks["raw_only minus raw_tp"] = source.load(HERE / "raw_only_evaluation.json")
    results_without_dead = {}
    for name, row in checks.items():
        a, b = name.split(" minus ")
        np.testing.assert_allclose(np.percentile(scores[a] - scores[b], [2.5, 97.5]), row["ci95"], atol=1e-12)
        left, drawn_left = sensitivity[a]
        right, drawn_right = sensitivity[b]
        results_without_dead[name] = dict(delta=left-right,
            ci95=np.percentile(drawn_left - drawn_right, [2.5, 97.5]).tolist())
    source.write("sensitivity.json", dict(excluded_class="Dead", comparisons=results_without_dead))
    source.write("validation.json", dict(passed=True, exact_solver_small_cases=True, patches=100,
                                          verified_comparisons=len(checks), fixed_masks=True))
    print("Checked exact solvers, mask preservation, shared Raw TP and paired intervals.")


if __name__ == "__main__":
    main()
