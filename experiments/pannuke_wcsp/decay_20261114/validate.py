"""Check split exclusions, recorded metrics and paired endpoints against artifacts."""
import hashlib

import numpy as np

from run import HERE, PREVIOUS, core, read, save


def main():
    split = read(HERE / "split.json")
    previous = read(PREVIOUS / "split.json")
    ids = [row["image_index"] for row in split["test"]]
    used = {row["image_index"] for group in ("train", "dev", "test") for row in previous[group]}
    assert len(ids) == len(set(ids)) == 100
    assert not (set(ids) & (used | set(split["excluded_ids"])))
    assert split["dev"] == previous["dev"]
    assert read(HERE / "v1_reproduction.json")["exact_prediction_equality"]
    frozen = read(HERE / "frozen.json")
    dev = read(HERE / "dev.json")
    for family, baseline in (("hybrid", dev[0]), ("potts", dev[1])):
        score = lambda row: row["biological_miou"]
        family_results = [row for row in dev if row["config"]["table"] == family]
        best = max(family_results, key=score)
        expected = baseline if score(best) - score(baseline) <= 0.00005 else best
        assert frozen[family] == expected["config"]
    final = read(HERE / "evaluation.json")
    inference = read(HERE / "inference.json")
    original_inference = read(PREVIOUS / "inference.json")
    for key in ("checkpoint_sha256", "architecture_sha256", "tiles", "input_padding", "fusion", "crop"):
        assert inference[key] == original_inference[key], key
    with np.load(HERE / "confusions.npz") as archive:
        matrices = {key: archive[key] for key in archive.files}
    for matrix in matrices.values():
        assert matrix.shape == (100, 6, 6)
        np.testing.assert_array_equal(matrix.sum((1, 2)), 256 * 256)
        np.testing.assert_array_equal(matrix.sum(2), matrices["raw"].sum(2))
    rng = np.random.default_rng(20261109)
    draws = rng.integers(len(ids), size=(5000, len(ids)))
    scores = {}
    sensitivity_scores = {}
    for name, cms in matrices.items():
        pooled = cms[draws].sum(1)
        tp = np.diagonal(pooled, axis1=1, axis2=2)
        union = pooled.sum(1) + pooled.sum(2) - tp
        ious = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)
        scores[name] = ious[:, 1:].mean(1)
        sensitivity_scores[name] = ious[:, [1, 2, 3, 5]].mean(1)
    outcomes = {}
    sensitivity = {}
    for label, pair in final["comparisons"].items():
        a, b = label.split(" minus ")
        observed = scores[a] - scores[b]
        np.testing.assert_allclose(np.percentile(observed, [2.5, 97.5]), pair["ci95"], atol=1e-12)
        outcomes[label] = observed
        which = [1, 2, 3, 5]
        observed_four = (np.array(core.metrics(matrices[a].sum(0))["iou"])[which]
                         - np.array(core.metrics(matrices[b].sum(0))["iou"])[which]).mean()
        sensitivity[label] = dict(delta=float(observed_four), ci95=np.percentile(
            sensitivity_scores[a] - sensitivity_scores[b], [2.5, 97.5]).tolist())
    for name, result in final["results"].items():
        assert [row["index"] for row in result["patches"]] == ids
        assert all(row["local_minimum"] for row in result["patches"])
        np.testing.assert_allclose(core.metrics(matrices[name].sum(0))["biological_miou"],
                                   result["biological_miou"], atol=1e-12)
        count = result["fixes"] + result["breaks"] + result["different_wrong"]
        assert 0 <= count <= len(ids) * 256 * 256
        assert result["fixes"] - result["breaks"] == (
            np.trace(matrices[name].sum(0)) - np.trace(matrices["raw"].sum(0))
        )
    np.savez_compressed(HERE / "bootstrap_replicates.npz", **outcomes)
    presence = (matrices["raw"].sum(2) > 0).sum(0).tolist()
    save("sensitivity.json", dict(excluded_class="Dead", patches_by_class=presence,
                                  comparisons=sensitivity))
    save("validation.json", dict(passed=True, n_eval=len(ids), comparisons=len(outcomes),
                                 source_hashes={name: hashlib.sha256((PREVIOUS / name).read_bytes()).hexdigest()
                                                for name in ("core.py", "tables.json", "split.json")}))
    print("Validated split, frozen selection, inference provenance, metrics and paired intervals.")


if __name__ == "__main__":
    main()
