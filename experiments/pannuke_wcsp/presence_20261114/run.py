"""Exact nucleus WCSP with a global type-set or type-count constraint."""
import argparse
import csv
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from prepare_fresh import CACHE, HERE, PREVIOUS, source

sys.path.insert(0, str(PREVIOUS))
import run as nucleus  # The existing experiment when this file is executed as __main__.

nucleus.CACHE = CACHE
N = 5
MASKS = np.arange(1, 1 << N)
ACTIVE = (MASKS[:, None] & (1 << np.arange(N))) != 0
BANDS = [8, 20, 40]
UNCERTAINTY = [.25, .5, 1., 2., 4.]


def fit(records):
    """Patch type sets estimated on annotations from the existing fitting split."""
    counts = np.zeros((len(BANDS) + 1, 1 << N), int)
    total_types = np.ones(N)
    for row in records:
        _, labels = nucleus.load_gt(row["image_index"])
        labels = labels[1:] - 1
        np.add.at(total_types, labels, 1)
        mask = np.bitwise_or.reduce(1 << labels.astype(int), initial=0)
        counts[np.searchsorted(BANDS, len(labels), side="right"), mask] += 1
    global_prob = (counts.sum(0) + 1) / (counts.sum() + counts.shape[1])
    banded = (counts + 16 * global_prob) / (counts.sum(1, keepdims=True) + 16)
    source.write("priors.json", dict(set_counts=counts.tolist(), global_probability=global_prob.tolist(),
                                    band_probability=banded.tolist(), type_frequency=(total_types / total_types.sum()).tolist()))
    return banded, total_types / total_types.sum()


def exact_set_options(unary):
    """Find the minimum summed unary subject to each exact nonempty type set."""
    n = len(unary)
    labels = np.zeros((len(MASKS), n), int)
    minima = np.full(len(MASKS), np.inf)
    for i, active in enumerate(ACTIVE):
        allowed = np.flatnonzero(active)
        if len(allowed) > n:
            continue
        options = unary[:, allowed]
        best = options.min(axis=1)
        label = allowed[options.argmin(axis=1)]
        # Every allowed type needs a distinct exemplar. Hungarian assignment
        # minimizes the additional cost of assigning those exemplars.
        kind, exemplar = linear_sum_assignment((options - best[:, None]).T)
        label[exemplar] = allowed[kind]
        labels[i] = label
        minima[i] = unary[np.arange(n), label].sum()
    return labels, minima


def exact_counts(unary, expected, strength):
    """Minimize unary plus separable convex penalties on predicted type counts."""
    n = len(unary)
    if n == 0:
        return np.empty(0, int)
    slots = np.arange(1, n + 1)
    increments = (2 * (slots - expected[:, None]) - 1) / (expected[:, None] + 1)
    extended = (unary[:, :, None] + strength * increments).reshape(n, N * n)
    rows, chosen = linear_sum_assignment(extended)
    assert np.array_equal(rows, np.arange(n))
    return chosen // n


def records(rows, band, frequency):
    data = list(nucleus.patches(rows))
    for patch in data:
        n = len(patch["baseline"])
        model_p = np.exp(-patch["cost"]["mean"])
        patch["set_probability"] = band[np.searchsorted(BANDS, n, side="right"), MASKS]
        patch["evidence_penalty"] = ACTIVE @ -np.log(np.maximum(model_p.max(0, initial=0), 1e-12))
        patch["options"] = {kind: exact_set_options(patch["cost"][kind]) for kind in ("vote", "mean", "log_mean")}
        patch["expected_model"] = model_p.sum(0)
        patch["expected_blend"] = .8 * patch["expected_model"] + .2 * frequency * n
    return data


def predict(config, patch):
    baseline = patch["baseline"]
    mode = config["mode"]
    if not len(baseline) or mode == "instance_vote":
        return baseline
    unary = patch["cost"][config["unary"]]
    strength = config["strength"]
    if mode == "unary":
        return unary.argmin(1)
    if mode == "pairwise":
        return nucleus.solve(unary, patch["graph"], 1 - np.eye(N), strength, baseline)
    if mode.startswith("count"):
        return exact_counts(unary, patch["expected_" + mode.split("_")[1]], strength)
    labels, minima = patch["options"][config["unary"]]
    cost = minima - strength * np.log(patch["set_probability"])
    if mode == "presence_evidence":
        cost = cost + strength * patch["evidence_penalty"]
    return labels[cost.argmin()]


def evaluate(config, data, save_types=False):
    matrices, overlaps, denominators = [], [], []
    changes = fixes = breaks = matched = 0
    for patch in data:
        labels = predict(config, patch)
        segmentation = np.r_[0, labels + 1][patch["instances"]]
        if save_types:
            np.testing.assert_array_equal(segmentation > 0, patch["instances"] > 0)
            directory = CACHE / "outputs" / config["name"]
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / f'{patch["index"]:04d}.npy', labels + 1)
        matrices.append(nucleus.confusion(patch["gt"], segmentation))
        truth, ids = patch["ts"], patch["ps"]
        before = patch["baseline"][ids] == truth
        after = labels[ids] == truth
        matched += int(after.sum())
        fixes += int((after & ~before).sum())
        breaks += int((before & ~after).sum())
        changes += int((labels != patch["baseline"]).sum())
        overlaps.append(np.bincount(truth[after], weights=patch["match_iou"][after], minlength=N))
        denominators.append((np.bincount(patch["truth"], minlength=N) + np.bincount(labels, minlength=N)) * .5)
    cm = np.stack(matrices)
    overlap, denom = np.array(overlaps), np.array(denominators)
    pq = np.divide(overlap.sum(0), denom.sum(0), out=np.zeros(N), where=denom.sum(0) > 0)
    return dict(config=config, bio_miou=float(nucleus.score(cm.sum(0))), typed_pq=float(pq.mean()),
                per_class_iou=nucleus.iou(cm.sum(0))[1:].tolist(), per_class_pq=pq.tolist(),
                changed_nuclei=changes, matched_correct=matched, fixes=fixes, breaks=breaks), cm, overlap, denom


def config(name, mode, unary="vote", strength=0):
    return dict(name=name, mode=mode, unary=unary, strength=strength)


def main(stage):
    source.HERE = HERE
    split = source.load(HERE / "split.json")
    control = config("vote", "instance_vote")
    pairwise = config("prior_potts", "pairwise", strength=2)
    grid = [config(f"{kind}_unary", "unary", kind) for kind in ("vote", "mean", "log_mean")]
    for kind in ("vote", "mean", "log_mean"):
        for mode in ("presence", "presence_evidence", "count_model", "count_blend"):
            for strength in UNCERTAINTY:
                grid.append(config(f"{kind}_{mode}_{strength:g}", mode, kind, strength))
    if stage == "dev":
        assert not (HERE / "frozen.json").exists()
        source.write("protocol.json", dict(
            primary="Pooled five-class mIoU vs. Raw TP; matched fixed-mask vote tests added value of WCSP.",
            development="112 previous instance patches explicitly reused for selection; 100 different patch IDs for final.",
            selection="Freeze the best type-set WCSP and best count WCSP on development mIoU. Within .0001 prefer smaller strength, then input grid order.",
            controls=[control, pairwise], grid=grid, threshold=.0001,
            inference="Existing frozen checkpoint, full 256x256 HoVer-Net masks and probabilities."
        ))
        band, frequency = fit(split["train"])
        data = records(split["dev"], band, frequency)
        table = [evaluate(candidate, data)[0] for candidate in [control, pairwise, *grid]]
        selected = {}
        for name in ("presence", "count"):
            family = [r for r in table if r["config"]["mode"].startswith(name)]
            maximum = max(r["bio_miou"] for r in family)
            close = [r for r in family if r["bio_miou"] >= maximum - .0001]
            leader = min(close, key=lambda row: row["config"]["strength"])
            selected[name] = leader["config"]
        source.write("development.json", table)
        source.write("frozen.json", selected)
        print("Frozen", selected, flush=True)
        for row in sorted(table, key=lambda r: r["bio_miou"], reverse=True)[:12]:
            print(row["config"]["name"], row["bio_miou"], row["fixes"], row["breaks"], flush=True)
        return

    assert not (HERE / "evaluation.json").exists()
    selected = source.load(HERE / "frozen.json")
    priors = source.load(HERE / "priors.json")
    data = records(split["test"], np.array(priors["band_probability"]), np.array(priors["type_frequency"]))
    raw = np.stack([nucleus.confusion(p["gt"], p["raw"]) for p in data])
    cms, overlap, denom, results = {"raw_tp": raw}, {}, {}, {}
    options = [control, pairwise, *selected.values()]
    for kind in sorted({r["unary"] for r in selected.values()} - {"vote"}):
        options.append(config(f"{kind}_unary", "unary", kind))
    for candidate in options:
        name = candidate["name"]
        results[name], cms[name], overlap[name], denom[name] = evaluate(candidate, data, save_types=True)
        print(name, results[name]["bio_miou"], results[name]["typed_pq"], flush=True)
    draws = np.random.default_rng(20261116).integers(len(data), size=(5000, len(data)))
    values = {name: nucleus.score(cm[draws].sum(1)) for name, cm in cms.items()}
    pq = {}
    for name in results:
        over, norm = overlap[name][draws].sum(1), denom[name][draws].sum(1)
        pq[name] = np.divide(over, norm, out=np.zeros_like(over), where=norm > 0).mean(1)
    pairs = [(name, "raw_tp") for name in results]
    for candidate in [pairwise, *selected.values()]:
        name = candidate["name"]
        pairs.append((name, "vote"))
        if candidate["unary"] != "vote":
            pairs.append((name, candidate["unary"] + "_unary"))
    comparisons = {}
    for a, b in pairs:
        delta = nucleus.score(cms[a]) - nucleus.score(cms[b])
        row = dict(delta=float(nucleus.score(cms[a].sum(0)) - nucleus.score(cms[b].sum(0))),
                   ci95=np.percentile(values[a] - values[b], [2.5, 97.5]).tolist(),
                   wlt=[int((delta > 1e-12).sum()), int((delta < -1e-12).sum()),
                        int((abs(delta) <= 1e-12).sum())],
                   per_class_delta=(nucleus.iou(cms[a].sum(0)) - nucleus.iou(cms[b].sum(0)))[1:].tolist())
        if b in pq:
            row["typed_pq_delta"] = results[a]["typed_pq"] - results[b]["typed_pq"]
            row["typed_pq_ci95"] = np.percentile(pq[a] - pq[b], [2.5, 97.5]).tolist()
        comparisons[f"{a} minus {b}"] = row
    np.savez_compressed(HERE / "confusions.npz", **cms)
    with (HERE / "per_patch.csv").open("w", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["image_index", *cms])
        for i, patch in enumerate(data):
            csv_writer.writerow([patch["index"], *[float(nucleus.score(matrix[i])) for matrix in cms.values()]])
    dead = sum(bool((p["truth"] == 3).any()) for p in data)
    source.write("evaluation.json", dict(raw_miou=float(nucleus.score(raw.sum(0))), results=results,
                                        comparisons=comparisons, n_patches=len(data), dead_patches=dead))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "final"])
    main(parser.parse_args().stage)
