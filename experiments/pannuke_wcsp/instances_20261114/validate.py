"""Independently reconstruct final pixel metrics and verify fixed masks."""
import numpy as np

from prepare import CACHE, HERE, load, write
from run import confusion, load_gt, score


def main():
    split = load(HERE / "split.json")
    result = load(HERE / "evaluation.json")
    frozen = load(HERE / "frozen.json")
    dev = load(HERE / "development.json")
    positive = [r for r in dev if r["config"].get("beta", 0) > 0]
    best = max(r["bio_miou"] for r in positive)
    kept = [r for r in positive if r["bio_miou"] >= best - .0001]
    chosen = min(kept, key=lambda r: (r["config"]["beta"], r["config"]["area_weight"],
                                      r["config"]["table"] != "potts"))
    assert chosen["config"] == frozen["wcsp"]
    ids = [r["image_index"] for r in split["test"]]
    assert ids == result["patch_ids"] and len(ids) == len(set(ids)) == 100
    assert not (set(ids) & set(split["excluded_ids"]))
    for group in ("train", "dev"):
        assert not (set(ids) & {r["image_index"] for r in split[group]})
    with np.load(HERE / "confusions.npz") as file:
        matrices = {key: file[key] for key in file.files}
    for i, index in enumerate(ids):
        gt_instances, types = load_gt(index)
        gt = types[gt_instances]
        with np.load(CACHE / f"pred_{index:04d}.npz") as file:
            instance_mask, raw = file["instances"], file["p"].argmax(-1)
        np.testing.assert_array_equal(confusion(gt, raw), matrices["raw_tp"][i])
        for name in result["rows"]:
            labels = np.load(CACHE / "refined" / name / f"{index:04d}.npy")
            assert len(labels) == instance_mask.max()
            assert np.all((labels >= 1) & (labels <= 5))
            semantic = np.r_[0, labels][instance_mask]
            np.testing.assert_array_equal(semantic > 0, instance_mask > 0)
            np.testing.assert_array_equal(confusion(gt, semantic), matrices[name][i])
    draws = np.random.default_rng(20261115).integers(100, size=(5000, 100))
    boot = {name: score(cm[draws].sum(1)) for name, cm in matrices.items()}
    for pair, item in result["comparisons"].items():
        a, b = pair.split(" minus ")
        np.testing.assert_allclose(
            np.percentile(boot[a] - boot[b], [2.5, 97.5]), item["ci95"], atol=1e-12)
    write("validation.json", dict(passed=True, patches=100, exact_fixed_masks=True,
                                  dev_selection=True, bootstrap_pairs=len(result["comparisons"])))
    print("Validated split, development selection, fixed masks, saved scores and bootstrap.")


if __name__ == "__main__":
    main()
