"""Select sibling schedules on development patches, then evaluate frozen models."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "hierarchy_20261109"
sys.path.insert(0, str(PREVIOUS))
import core


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def candidate(name, beta, table="hybrid", gamma=0):
    return dict(name=name, beta=list(beta), table=table, gamma=gamma, parent=0.1)


def prepare_fresh():
    import importlib.util
    spec = importlib.util.spec_from_file_location("full_patch_inference", PREVIOUS / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module.old_module()
    original = read(PREVIOUS / "split.json")
    known = set(module.exclusions()) | {str(i) for i in old.exclusions()}
    known.update(str(r["image_index"]) for group in ("train", "dev", "test") for r in original[group])
    audit = read(PREVIOUS / "upstream_overlap_audit.json")
    known.update(map(str, audit["comparison_overlap"]))
    if not (HERE / "split.json").exists():
        archive, member = old.members("types.npy")
        with archive, archive.open(member) as file:
            tissues = np.load(file)
        eligible = [i for i in range(len(tissues)) if str(i) not in known]
        ids = np.random.default_rng(20261114).choice(eligible, 100, replace=False)
        records = [dict(image_index=int(i), tissue=str(tissues[i])) for i in sorted(ids)]
        save("split.json", dict(dev=original["dev"], test=records, eligible_count=len(eligible),
                               seed=20261114, excluded_ids=sorted(map(int, known)),
                               selection="Uniform over eligible fold-2 patch IDs, without prediction or label selection. Patient and checkpoint independence unverified."))
    ids = [r["image_index"] for r in read(HERE / "split.json")["test"]]
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    for source, prefix in (("masks.npy", "gt"), ("images.npy", "rgb")):
        missing = [i for i in ids if not (cache / f"{prefix}_{i:04d}.npy").exists()]
        if not missing:
            continue
        for i, array in old.stream_arrays(source, missing):
            if prefix == "gt":
                mask = np.zeros((256, 256), np.uint8)
                for k in range(5):
                    mask[array[..., k] > 0] = k + 1
                array = mask
            converted = array.astype(np.uint8)
            assert np.array_equal(array, converted), (i, prefix)
            np.save(cache / f"{prefix}_{i:04d}.npy", converted)
        print("Extracted", prefix, flush=True)
    save("prepared.json", {"complete": True})
    # Reuse the verified full-patch extraction, checkpoint and overlap-fusion path.
    module.HERE = HERE
    split = read(HERE / "split.json")
    for row in split["dev"]:
        i = row["image_index"]
        target = cache / f"prob_{i:04d}.npz"
        if not target.exists():
            target.symlink_to(PREVIOUS / "cache" / target.name)
    module.infer()


def panels(records, cache):
    for record in records:
        index = record["image_index"]
        with np.load(cache / f"prob_{index:04d}.npz") as file:
            p = file["p"].astype(float)
        gt = np.load(cache / f"gt_{index:04d}.npy")
        yield index, p, gt


def evaluate(config, patches, tables, graph, reference=False, save_labels=False):
    name = config["name"]
    weights = np.array(config["beta"])
    costs = tables[config["table"]]
    raw_matrices, matrices, rows = [], [], []
    level_changes = np.zeros(6, int)
    level_sizes = np.zeros(6, int)
    for index, p, gt in patches:
        raw = p.argmax(-1)
        # Multiplying the six tables supplies per-level beta to the existing
        # solver. Passing scalar 1 preserves its degree-of-four normalization.
        labels, info = core.solve(p, costs * weights[:, None, None], graph,
                                  1, config["parent"], config["gamma"])
        pred = labels[:gt.size].reshape(gt.shape)
        if reference:
            with np.load(PREVIOUS / "version_1" / f"patch_{index:04d}.npz") as file:
                np.testing.assert_array_equal(pred, file["pred"])
                if "levels" in file:
                    np.testing.assert_array_equal(labels, file["levels"])
        if "margin_gate" in config:
            pred = gate(p, pred, config["margin_gate"])
        if save_labels:
            dest = HERE / "predictions" / name
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / f"{index:04d}.npz", levels=labels, pred=pred)
        raw_matrices.append(core.confusion(gt, raw))
        matrices.append(core.confusion(gt, pred))
        changes = raw != pred
        fixes = np.count_nonzero((raw != gt) & (pred == gt))
        breaks = np.count_nonzero((raw == gt) & (pred != gt))
        rows.append(dict(index=index, fixes=fixes, breaks=breaks,
                         different_wrong=int(changes.sum()) - fixes - breaks,
                         rounds=info["rounds"], seconds=info["seconds"],
                         local_minimum=info["verified_local_minimum"]))
        level_changes += [r["changed"] for r in info["levels"]]
        level_sizes += [r["variables"] for r in info["levels"]]
    raw_matrices, matrices = np.array(raw_matrices), np.array(matrices)
    delta = np.array([core.metrics(y)["biological_miou"] - core.metrics(x)["biological_miou"]
                      for x, y in zip(raw_matrices, matrices)])
    aggregate = {key: sum(row[key] for row in rows)
                 for key in ("fixes", "breaks", "different_wrong", "rounds", "seconds")}
    result = dict(config=config, **core.metrics(matrices.sum(0)), **aggregate,
                  raw=core.metrics(raw_matrices.sum(0)), level_change_rate=(level_changes / level_sizes).tolist(),
                  wlt=[int((delta > 1e-12).sum()), int((delta < -1e-12).sum()),
                       int((abs(delta) <= 1e-12).sum())], patches=rows)
    assert all(row["local_minimum"] for row in rows), name
    print(name, f'{result["biological_miou"]:.8f}', result["wlt"], flush=True)
    return result, raw_matrices, matrices


def gate(probabilities, refined, threshold):
    """Accept a WCSP edit only at pixels with a small top-two margin."""
    top = np.partition(probabilities, -2, axis=-1)[..., -2:]
    margin = top[..., 1] - top[..., 0]
    return np.where(margin <= threshold, refined, probabilities.argmax(-1)).astype(np.uint8)


def select_gate(frozen, tables, graph):
    thresholds = [.01, .025, .05, .10]
    records = read(PREVIOUS / "split.json")["dev"]
    rows = []
    bases = [frozen["hybrid"], frozen["potts"]]
    dev = {row["config"]["name"]: row for row in read(HERE / "dev.json")}
    best_ungated = max(dev[base["name"]]["biological_miou"] for base in bases)
    for base in bases:
        blocks = np.array(base["beta"])[:, None, None]
        weighted = tables[base["table"]] * blocks
        matrices = np.zeros((len(thresholds), 6, 6), dtype=np.int64)
        fixes = np.zeros(len(thresholds), dtype=int)
        breaks = np.zeros(len(thresholds), dtype=int)
        for _, p, gt in panels(records, PREVIOUS / "cache"):
            labels, _ = core.solve(p, weighted, graph, 1, base["parent"], base["gamma"])
            raw = p.argmax(-1)
            pred = labels[:gt.size].reshape(gt.shape)
            for i, threshold in enumerate(thresholds):
                output = gate(p, pred, threshold)
                matrices[i] += core.confusion(gt, output)
                fixes[i] += np.count_nonzero((raw != gt) & (output == gt))
                breaks[i] += np.count_nonzero((raw == gt) & (output != gt))
        for threshold, matrix, fix, brk in zip(thresholds, matrices, fixes, breaks):
            config = dict(base, name=f'{base["name"]}_gate_{threshold}', margin_gate=threshold)
            row = dict(config=config, **core.metrics(matrix), fixes=int(fix), breaks=int(brk))
            rows.append(row)
            print(config["name"], row["biological_miou"], int(fix), int(brk), flush=True)
    best = max(rows, key=lambda row: row["biological_miou"])
    selected = best["config"] if best["biological_miou"] - best_ungated > .00005 else None
    save("gate.json", dict(candidates=rows, best_ungated_bio_miou=best_ungated, selected=selected,
                           rule="Advance only if gated dev bio mIoU exceeds the best frozen ungated variant by more than 0.00005."))


def check_archived(decay, tables, graph):
    """Probe the same frozen schedule on the already inspected comparison set."""
    records = read(PREVIOUS / "split.json")["test"]
    matrices = {}
    for name, path in (
        ("v1", PREVIOUS / "version_1" / "confusions.npz"),
        ("potts", HERE.parent / "scaling_siblings_20261109" / "potts" / "confusions.npz"),
    ):
        with np.load(path) as file:
            if "raw" in matrices:
                np.testing.assert_array_equal(matrices["raw"], file["raw"])
            matrices["raw"] = file["raw"]
            matrices[name] = file["wcsp"]
    runs = {}
    for config in [decay, dict(decay, name="matched_potts", table="potts")]:
        result, raw, output = evaluate(config, panels(records, PREVIOUS / "cache"), tables, graph)
        np.testing.assert_array_equal(matrices["raw"], raw)
        matrices[config["name"]] = output
        runs[config["name"]] = result
    metrics = {name: core.metrics(m.sum(0)) for name, m in matrices.items()}
    pairs = {}
    for a, b in [(decay["name"], "v1"), (decay["name"], "matched_potts"), ("matched_potts", "potts")]:
        pairs[f"{a} minus {b}"] = dict(
            delta=metrics[a]["biological_miou"] - metrics[b]["biological_miou"],
            ci95=core.bootstrap(matrices[b], matrices[a]))
    np.savez_compressed(HERE / "archived_confusions.npz", **matrices)
    save("archived.json", dict(
        use="Exploratory check on the previously inspected 100 patches after freezing selection.",
        dead_positive_patches=int((matrices["raw"].sum(2)[:, 4] > 0).sum()),
        metrics=metrics, comparisons=pairs, runs=runs))


def main(stage):
    tables = {"hybrid": np.array(read(PREVIOUS / "tables.json")["final"]),
              "potts": np.repeat((1 - np.eye(6))[None, :, :], 6, axis=0)}
    graph = core.graph()
    flat = candidate("v1", [.3] * 6)
    potts = candidate("potts", [.3] * 6, "potts")
    opts = [flat, potts]
    for d in (.8, .6, .4, .2):
        schedule = [.3 * d ** i for i in range(5, -1, -1)]
        opts.extend([candidate(f"blocks_{d}", [.3] + schedule[1:]),
                     candidate(f"all_{d}", schedule),
                     candidate(f"pixels_{d}", schedule[:1] + [.3] * 5),
                     candidate(f"potts_blocks_{d}", [.3] + schedule[1:], "potts")])
    for strength in (.075, .15, .225):
        opts.append(candidate(f"uniform_blocks_{strength}", [.3] + [strength] * 5))
    for strength in (.45, .6):
        opts.append(candidate(f"coarse_boost_{strength}", [.3] * 4 + [strength] * 2))
    if stage == "dev":
        assert not (HERE / "frozen.json").exists()
        records = read(PREVIOUS / "split.json")["dev"]
        patches = list(panels(records, PREVIOUS / "cache"))
        save("protocol.json", dict(dev=records, candidates=opts,
             objective="V1 solver with six sibling coefficients, parent 0.10; V2 cross-check gamma 0.30.",
             selection="Max development pooled biological mIoU. Prefer baseline within 0.00005 absolute mIoU; otherwise maximize and break exact ties by candidate order. Try V2 on V1 and highest-scoring nonuniform hybrid decay. Final: one hybrid and one tuned Potts at most plus their controls."))
        rows = [evaluate(c, patches, tables, graph)[0] for c in opts]
        score = lambda row: row["biological_miou"]
        decayed = [r for r in rows if r["config"]["name"].startswith(("all_", "blocks_"))]
        best_decay = max(decayed, key=score)["config"]
        for config in (flat, best_decay):
            rows.append(evaluate(dict(config, name="v2_" + config["name"], gamma=.3),
                                 patches, tables, graph)[0])
        hybrid = max((r for r in rows if r["config"]["table"] == "hybrid"), key=score)
        generic = max((r for r in rows if r["config"]["table"] == "potts"), key=score)
        hybrid = rows[0] if score(hybrid) - score(rows[0]) <= .00005 else hybrid
        generic = rows[1] if score(generic) - score(rows[1]) <= .00005 else generic
        save("dev.json", rows)
        save("frozen.json", dict(hybrid=hybrid["config"], potts=generic["config"],
                                 best_nonuniform_hybrid_decay=best_decay))
        return
    if stage == "audit":
        check = [evaluate(flat, panels([r], PREVIOUS / "cache"), tables, graph, reference=True)[0]
                 for r in read(PREVIOUS / "split.json")["test"]]
        save("v1_reproduction.json", dict(patches=len(check), exact_prediction_equality=True))
        return
    frozen = read(HERE / "frozen.json")
    if stage == "gate":
        select_gate(frozen, tables, graph)
        return
    if stage == "archive":
        check_archived(frozen["best_nonuniform_hybrid_decay"], tables, graph)
        return
    assert not (HERE / "evaluation.json").exists()
    hybrid = frozen["hybrid"]
    decay = frozen["best_nonuniform_hybrid_decay"]
    configs = [flat, potts, hybrid, frozen["potts"], decay,
               dict(decay, name="matched_potts", table="potts")]
    gate_choice = read(HERE / "gate.json")["selected"] if (HERE / "gate.json").exists() else None
    if gate_choice:
        configs.append(gate_choice)
    by_name = {c["name"]: c for c in configs}
    records = read(HERE / "split.json")["test"]
    results, matrices = {}, {}
    for name, config in by_name.items():
        result, raw, output = evaluate(config, panels(records, HERE / "cache"),
                                       tables, graph, save_labels=True)
        results[name] = result
        matrices[name] = output
    matrices["raw"] = raw
    np.savez_compressed(HERE / "confusions.npz", **matrices)
    pairs = [(name, "raw") for name in results]
    pairs.extend([("potts", "v1"), ("matched_potts", "potts")])
    pairs.extend((decay["name"], control) for control in ("v1", "potts", "matched_potts"))
    if hybrid["name"] not in {"v1", decay["name"]}:
        pairs.extend((hybrid["name"], control) for control in ("v1", "potts"))
    if frozen["potts"]["name"] != "potts":
        pairs.append((frozen["potts"]["name"], "potts"))
    if gate_choice:
        base_name = gate_choice["name"].split("_gate_", 1)[0]
        pairs.append((gate_choice["name"], base_name))
    comparisons = {}
    for a, b in pairs:
        ma, mb = (core.metrics(matrices[k].sum(0)) for k in (a, b))
        comparisons[f"{a} minus {b}"] = dict(
            delta=ma["biological_miou"] - mb["biological_miou"],
            ci95=core.bootstrap(matrices[b], matrices[a]),
            per_class_delta=(np.array(ma["iou"]) - mb["iou"]).tolist())
    save("evaluation.json", dict(results=results, comparisons=comparisons))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "prepare", "audit", "gate", "archive", "evaluate"])
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare_fresh()
    else:
        main(stage)
