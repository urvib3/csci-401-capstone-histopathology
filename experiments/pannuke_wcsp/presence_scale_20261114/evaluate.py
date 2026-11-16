"""Evaluate frozen presence WCSP on the random and Dead-positive cohorts."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

from prepare_scale import CACHE, HERE, PREVIOUS, SEED, WORKERS, source


def load_previous():
    name = "previous_presence_wcsp"
    spec = importlib.util.spec_from_file_location(name, PREVIOUS / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_weights(n, seed):
    draws = np.random.default_rng(seed).integers(n, size=(5000, n))
    counts = np.zeros((len(draws), n), dtype=np.int32)
    np.add.at(counts, (np.arange(len(draws))[:, None], draws), 1)
    return counts


def interval(samples):
    return np.percentile(samples, [2.5, 97.5], axis=0).tolist()


def prediction_audit(presence, cohort, data, winner, matrices):
    for i, patch in enumerate(data):
        ids = np.load(CACHE / "outputs" / winner / f'{patch["index"]:04d}.npy')
        mask = patch["instances"]
        assert len(ids) == mask.max() and np.all((ids >= 1) & (ids <= 5))
        image = np.r_[0, ids][mask]
        np.testing.assert_array_equal(mask > 0, image > 0)
        np.testing.assert_array_equal(presence.nucleus.confusion(patch["gt"], image), matrices[winner][i])
    print(cohort, "verified label assignments and fixed masks", flush=True)


def evaluate(presence, name, records, winner, priors):
    data = presence.records(records, np.array(priors["band_probability"]), np.array(priors["type_frequency"]))
    raw = np.stack([presence.nucleus.confusion(patch["gt"], patch["raw"]) for patch in data])
    matrices, results, overlaps, denominators = {"raw_tp": raw}, {}, {}, {}
    options = [presence.config("vote", "instance_vote"), winner]
    for candidate in options:
        key = candidate["name"]
        results[key], matrices[key], overlaps[key], denominators[key] = presence.evaluate(candidate, data, save_types=True)
    winner_name = winner["name"]
    prediction_audit(presence, name, data, winner_name, matrices)
    weights = bootstrap_weights(len(data), SEED)
    boot_iou = {key: presence.nucleus.iou((weights @ cm.reshape(len(data), -1)).reshape(-1, 6, 6))
                for key, cm in matrices.items()}
    points_iou = {key: presence.nucleus.iou(cm.sum(0)) for key, cm in matrices.items()}
    boot_pq = {}
    for key in results:
        total_overlap, denominator = weights @ overlaps[key], weights @ denominators[key]
        boot_pq[key] = np.divide(total_overlap, denominator, out=np.zeros_like(total_overlap),
                                 where=denominator > 0).mean(1)
    comparisons = {}
    for a, b in [("vote", "raw_tp"), (winner_name, "raw_tp"), (winner_name, "vote")]:
        diff = boot_iou[a] - boot_iou[b]
        point = points_iou[a] - points_iou[b]
        pdiff = presence.nucleus.score(matrices[a]) - presence.nucleus.score(matrices[b])
        selected_classes = [1, 2, 3, 5]
        row = dict(delta=float(point[1:].mean()), ci95=interval(diff[:, 1:].mean(1)),
                   no_dead_delta=float(point[selected_classes].mean()),
                   no_dead_ci95=interval(diff[:, selected_classes].mean(1)),
                   class_delta=point[1:].tolist(), class_ci95=np.array(interval(diff[:, 1:])).T.tolist(),
                   wlt=[int((pdiff > 1e-12).sum()), int((pdiff < -1e-12).sum()),
                        int((abs(pdiff) <= 1e-12).sum())])
        if b in results:
            row.update(typed_pq_delta=results[a]["typed_pq"] - results[b]["typed_pq"],
                       typed_pq_ci95=interval(boot_pq[a] - boot_pq[b]))
        comparisons[f"{a} minus {b}"] = row
    truth_by_patch = np.stack([np.bincount(patch["truth"], minlength=5) for patch in data])
    np.savez_compressed(HERE / f"{name}_confusions.npz", **matrices)
    with (HERE / f"{name}_patches.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_index", "tissue", "dead_nuclei", "matched_nuclei", *matrices])
        for i, patch in enumerate(data):
            writer.writerow([patch["index"], records[i]["tissue"], truth_by_patch[i, 3], len(patch["ps"]),
                             *[float(presence.nucleus.score(cm[i])) for cm in matrices.values()]])
    matches = sum(len(patch["ps"]) for patch in data)
    return dict(n_patches=len(data), matches=matches, detected=sum(len(p["baseline"]) for p in data),
                nucleus_counts=truth_by_patch.sum(0).tolist(), patch_support=(truth_by_patch > 0).sum(0).tolist(),
                raw_miou=float(points_iou["raw_tp"][1:].mean()), raw_iou=points_iou["raw_tp"][1:].tolist(),
                results=results, comparisons=comparisons)


def main():
    source.HERE = HERE
    split = source.load(HERE / "split.json")
    protocol = source.load(HERE / "protocol.json")
    for filename, expected in protocol["frozen_inputs"].items():
        assert hashlib.sha256((PREVIOUS / filename).read_bytes()).hexdigest() == expected
    primary = {r["image_index"] for r in split["primary"]}
    secondary = {r["image_index"] for r in split["dead_stress"]}
    assert len(primary) == 300 and len(secondary) == protocol["n_dead_stress"]
    assert not primary & secondary and not set(split["excluded_ids"]) & (primary | secondary)
    if (HERE / "inference_test.json").exists():
        inference_receipts = [source.load(HERE / "inference_test.json")]
        backends = source.load(HERE / "inference_backends.json")
        assert set(backends["cpu_patch_ids"]).isdisjoint(backends["mps_patch_ids"])
        assert set(backends["cpu_patch_ids"] + backends["mps_patch_ids"]) == primary | secondary
    else:
        inference_receipts = [source.load(HERE / f"inference_{i}.json") for i in range(WORKERS)]
        backends = dict(cpu_patch_ids=sorted(primary | secondary), mps_patch_ids=[])
    reference = source.load(PREVIOUS / "inference_test.json")
    coverage = []
    for receipt in inference_receipts:
        coverage += receipt["patch_ids"]
        for key in set(reference) - {"patch_ids"}:
            assert receipt[key] == reference[key], key
    assert set(coverage) == primary | secondary and len(set(coverage)) == len(coverage)
    presence = load_previous()
    presence.CACHE = CACHE
    presence.nucleus.CACHE = CACHE
    winner = source.load(PREVIOUS / "frozen.json")["presence"]
    priors = source.load(PREVIOUS / "priors.json")
    results = {name: evaluate(presence, name, split[name], winner, priors)
               for name in ("primary", "dead_stress") if split[name]}
    source.write("evaluation.json", dict(config=winner, verified=True, cohorts=results, backends={k: len(v) for k, v in backends.items()}))
    for name, result in results.items():
        print(name, result["n_patches"], "Raw TP", result["raw_miou"], flush=True)
        for label, comparison in result["comparisons"].items():
            print(label, comparison, flush=True)


if __name__ == "__main__":
    main()
