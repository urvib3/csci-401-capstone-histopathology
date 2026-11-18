"""Separate hierarchy weighting schedules, with frozen paired comparisons."""
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIER = ROOT / "hierarchy_20261109"
sys.path.insert(0, str(HIER))
import core

SIZES = np.array(core.SIZES, dtype=float)
GRAPH = core.graph()
POTTS = 1 - np.eye(6)
HYBRID = np.array(json.loads((HIER / "tables.json").read_text())["final"])
TABLES = {"hybrid": HYBRID, "potts": np.repeat(POTTS[None], 6, axis=0)}
SEED = 20261117


def read(path):
    return json.loads(path.read_text())


def write(file, value):
    path = HERE / file
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def settings(name, beta=None, parent=None, gamma=None, adaptive=False, table="hybrid"):
    return dict(name=name, beta=[.3] * 6 if beta is None else list(beta),
                parent=[.1] * 5 if parent is None else list(parent),
                gamma=[0.] * 5 if gamma is None else list(gamma),
                adaptive=adaptive, table=table)


def schedule(sides, power, mean):
    raw = sides ** power
    return mean * raw / raw.mean()


def cache_inputs(directory, rows, summed_inputs=True):
    for row in rows:
        i = row["image_index"]
        path = directory / f"pred_{i:04d}.npz"
        if directory == HIER / "cache":
            with np.load(directory / f"prob_{i:04d}.npz") as data:
                probabilities = data["p"].astype(float)
            gt = np.load(directory / f"gt_{i:04d}.npy")
        else:
            with np.load(path) as data:
                probabilities = data["p"].astype(float)
            with np.load(directory / f"gt_{i:04d}.npz") as data:
                gt = data["types"][data["instances"]]
        start, unary = core.initial_unary(probabilities, 1.)
        levels = [slice(a, b) for a, b in zip(GRAPH["offsets"], GRAPH["offsets"][1:])]
        margins = [np.partition(p, -2, axis=-1)[..., -2:].reshape(-1, 2)
                   for p in core.block_probabilities(probabilities)[1:]]
        adaptive = [.25 + .75 * (values[:, 1] - values[:, 0]) for values in margins]
        yield i, probabilities if summed_inputs else None, gt, start, unary, levels, adaptive


def evaluate_patch(config, inputs):
    _, _, gt, initial, common_unary, levels, adaptive = inputs
    y = initial.copy()
    unary = common_unary.copy()
    for level, (section, weight) in enumerate(zip(levels[1:], config["gamma"])):
        factor = weight * adaptive[level][:, None] if config["adaptive"] else weight
        unary[section] *= factor
    sibling = TABLES[config["table"]] * np.array(config["beta"])[:, None, None] / 4
    parent = POTTS[None] * np.array(config["parent"])[:, None, None]
    costs = np.concatenate((sibling, parent))
    previous = core.energy(y, unary, costs, GRAPH)
    for rounds in range(1, 101):
        n = sum(core.sweep(y, unary, costs, GRAPH["ptr"], GRAPH["dst"], GRAPH["typ"], rev)
                for rev in (False, True))
        current = core.energy(y, unary, costs, GRAPH)
        assert current <= previous + 1e-7
        previous = current
        if n == 0:
            break
    assert not n and core.audit(y, unary, costs, GRAPH)["remaining_improving_variables"] == 0
    pred = y[:gt.size].reshape(gt.shape)
    return pred, rounds


def evaluate(configs, cache_dir, rows, verify_ids=()):
    cm = {c["name"]: [] for c in configs} | {"raw": []}
    stats = {c["name"]: dict(fixed=0, broken=0, seconds=0., rounds=0) for c in configs}
    for j, patch in enumerate(cache_inputs(cache_dir, rows, bool(verify_ids))):
        i, p, gt, initial, *_ = patch
        raw = initial[:gt.size].reshape(gt.shape)
        cm["raw"].append(core.confusion(gt, raw))
        for config in configs:
            name = config["name"]
            start = time.perf_counter()
            pred, rounds = evaluate_patch(config, patch)
            stats[name]["seconds"] += time.perf_counter() - start
            stats[name]["rounds"] += rounds
            cm[name].append(core.confusion(gt, pred))
            stats[name]["fixed"] += int(((raw != gt) & (pred == gt)).sum())
            stats[name]["broken"] += int(((raw == gt) & (pred != gt)).sum())
            if i in verify_ids and name in {"v1", "v2"}:
                old, _ = core.solve(p, HYBRID, GRAPH, .3, .1, 0 if name == "v1" else .3)
                np.testing.assert_array_equal(old[:gt.size].reshape(gt.shape), pred)
        if (j + 1) % 30 == 0 or j + 1 == len(rows):
            print(f"{j + 1}/{len(rows)} patches, {len(configs)} configurations", flush=True)
    cm = {name: np.stack(mats) for name, mats in cm.items()}
    summary = {name: dict(core.metrics(mats.sum(0)), **stats.get(name, {}))
               for name, mats in cm.items()}
    return summary, cm


def grid():
    arms = [settings("v1"), settings("v2", gamma=[.3] * 5),
            settings("old_decay", beta=[.3, .03888, .0648, .108, .18, .3]),
            settings("potts", table="potts")]
    for power in (-1, -.5, .5, 1):
        arms.append(settings(f"parent_{power:g}", parent=schedule(SIZES[:5], power, .1)))
        arms.append(settings(f"sibling_{power:g}", beta=[.3, *schedule(SIZES[1:], power, .3)]))
        arms.append(settings(f"unary_{power:g}", gamma=schedule(SIZES[1:], power, .3)))
    arms += [settings("parent_area", parent=schedule(SIZES[:5], 2, .1)),
             settings("parent_low", parent=[.03] * 5), settings("parent_high", parent=[.3] * 5),
             settings("sibling_low", beta=[.3] + [.075] * 5),
             settings("unary_low", gamma=[.15] * 5),
             settings("unary_adaptive", gamma=[.3] * 5, adaptive=True),
             settings("unary_adaptive_fine", gamma=schedule(SIZES[1:], -.5, .3), adaptive=True),
             settings("unary_adaptive_coarse", gamma=schedule(SIZES[1:], .5, .3), adaptive=True)]
    return arms


def source_splits():
    dev = read(HIER / "split.json")["dev"]
    extra = read(ROOT / "instances_20261114/split.json")["test"]
    rng = np.random.default_rng(SEED)
    extra = [extra[i] for i in sorted(rng.choice(len(extra), 36, replace=False))]
    holdout = read(ROOT / "presence_scale_20261114/split.json")["primary"]
    assert len(holdout) == 300 and not ({r["image_index"] for r in dev + extra} &
                                       {r["image_index"] for r in holdout})
    return dev, extra, holdout


def compare(cm, pairs):
    rng = np.random.default_rng(SEED)
    n = len(cm["raw"])
    draws = rng.integers(n, size=(5000, n))
    weight = np.zeros((len(draws), n), dtype=np.int32)
    np.add.at(weight, (np.arange(len(draws))[:, None], draws), 1)

    def iou(x):
        tp = np.diagonal(x, axis1=-2, axis2=-1)
        union = x.sum(-1) + x.sum(-2) - tp
        return np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)

    io = {name: iou((weight @ mat.reshape(n, -1)).reshape(-1, 6, 6)) for name, mat in cm.items()}
    out = {}
    for a, b in pairs:
        points = iou(cm[a].sum(0)) - iou(cm[b].sum(0))
        boot = io[a] - io[b]
        patch = iou(cm[a])[:, 1:].mean(1) - iou(cm[b])[:, 1:].mean(1)
        other = [1, 2, 3, 5]
        out[f"{a} minus {b}"] = dict(
            delta=float(points[1:].mean()), ci95=np.percentile(boot[:, 1:].mean(1), [2.5, 97.5]).tolist(),
            per_class_delta=points[1:].tolist(), no_dead_delta=float(points[other].mean()),
            no_dead_ci95=np.percentile(boot[:, other].mean(1), [2.5, 97.5]).tolist(),
            wlt=[int((patch > 1e-12).sum()), int((patch < -1e-12).sum()), int((abs(patch) <= 1e-12).sum())])
    return out


def main():
    assert not (HERE / "evaluation.json").exists(), "Completed output already exists."
    dev, extra, holdout = source_splits()
    arms = grid()
    write("protocol.json", dict(original_dev=dev, additional_dev=extra, comparison=holdout, one_family=arms,
          selection="Within each family select highest dev pooled mIoU with stable ties. Test the combined schedule and its three individual families.",
          scope="Exploratory transfer to 300 existing type-set comparison patches. No hierarchy settings fitted on these IDs. Prior type-set outcomes are already known.",
          family_weights="Block-only sibling schedule preserves pixel beta. Graded schedules hold arithmetic mean coefficient over affected levels fixed, not edge-count-weighted.",
          solver="Original bidirectional ICM sweeps and objective audit. No global optimality claim."))
    if (HERE / "frozen.json").exists():
        selected = read(HERE / "frozen.json")["configurations"]
    else:
        dev_caches = [(HIER / "cache", dev), (ROOT / "instances_20261114/cache", extra)]
        per_source = [evaluate(arms, cache, rows, [r["image_index"] for r in rows[:2]] if j == 0 else ())[1]
                      for j, (cache, rows) in enumerate(dev_caches)]
        score = {name: core.metrics(sum(m[name].sum(0) for m in per_source))["biological_miou"]
                 for name in per_source[0]}
        leaders = {family: max((c for c in arms if c["name"].startswith(family + "_")),
                               key=lambda c: score[c["name"]]) for family in ("parent", "sibling", "unary")}
        mixed = settings("combined", beta=leaders["sibling"]["beta"], parent=leaders["parent"]["parent"],
                         gamma=leaders["unary"]["gamma"], adaptive=leaders["unary"]["adaptive"])
        selected = [*arms[:4], *leaders.values(), mixed]
        write("frozen.json", dict(configurations=selected, development_miou=score, leaders=leaders))
        print("Frozen", [(c["name"], score.get(c["name"])) for c in selected], flush=True)
    summary, matrices = evaluate(selected, ROOT / "presence_scale_20261114/cache", holdout)
    pairs = [(c["name"], "raw") for c in selected]
    for cfg in selected:
        name = cfg["name"]
        if name not in {"v1", "v2"}:
            pairs += [(name, "v1")]
        if name.startswith("unary") or name == "combined":
            pairs += [(name, "v2")]
    fixed = read(ROOT / "presence_scale_20261114/evaluation.json")["cohorts"]["primary"]
    with np.load(ROOT / "presence_scale_20261114/primary_confusions.npz") as archive:
        np.testing.assert_array_equal(matrices["raw"], archive["raw_tp"])
        matrices["frozen_type_set"] = archive["vote_presence_evidence_1"]
    summary["frozen_type_set"] = core.metrics(matrices["frozen_type_set"].sum(0))
    pairs += [("frozen_type_set", "raw"), ("frozen_type_set", "combined")]
    for cfg in selected[4:7]:
        pairs.append(("frozen_type_set", cfg["name"]))
    comparisons = compare(matrices, pairs)
    np.savez_compressed(HERE / "confusions.npz", **matrices)
    write("evaluation.json", dict(summary=summary, comparisons=comparisons,
                                  type_set_reference=fixed["comparisons"]["vote_presence_evidence_1 minus raw_tp"],
                                  local_optimum_audit_passed=True))
    for name, row in comparisons.items():
        print(name, f'{row["delta"]:+.7f}', row["ci95"], flush=True)


if __name__ == "__main__":
    main()
