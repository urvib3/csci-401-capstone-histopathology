# The steps behind `python -m wcsp ...`, usable from Python too.
#
#   predict    run the pixel classifier once and cache its probabilities
#   calibrate  fit the classifier's temperature on cached dev predictions
#   solve      build and solve the WCSP per sample, score against ground truth
#   summarize  merge the per-shard results of a run into summary tables

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .classifiers.cache import ProbStore
from .data import load_dataset, select
from .evaluate.compare import contrast_bootstrap, paired_bootstrap
from .evaluate.nuclei import instance_classes, nucleus_counts, nucleus_report
from .evaluate.pixel import pixel_confusion, pixel_report
from .evaluate.rules import pair_counts, rule_report
from .labels import IGNORE
from .model.problem import build_problem
from .solve.argmax import ArgmaxSolver
from .solve.energy import energy
from .solve.factory import get_solver


def dataset_for(cfg, split=None):
    spec = dict(cfg.dataset)
    name = spec.pop("name")
    spec.pop("split", None)
    return load_dataset(name, split or cfg.dataset["split"], **spec)


def store_for(cfg, split):
    return ProbStore(cfg.cache_root, cfg.dataset["name"], cfg.classifier["name"], split)


def classifier_for(cfg):
    spec = dict(cfg.classifier)
    name = spec.pop("name")
    spec.pop("temperature", None)
    if name in ("tia_unet", "tia_unet_native"):   # separate names keep separate caches
        from .classifiers.tia_unet import TiaUNet

        return TiaUNet(native=name == "tia_unet_native", **spec)
    if name == "tia_unet_fullres":
        from .classifiers.tia_unet import FullResTiaUNet

        return FullResTiaUNet(**spec)
    raise ValueError(f"unknown classifier {name!r}.")


def _scored(sample):
    """True if the grid has at least one labeled pixel (others add nothing to any metric)."""
    target = sample.grid_target
    return target is None or bool((target != IGNORE).any())


def predict(cfg, split, shard=(0, 1), batch_size=8, limit=0):
    data = dataset_for(cfg, split)
    store = store_for(cfg, split)
    index, count = shard
    path = store.shard_path(index, count)
    if path.exists():
        print(f"{path} exists; skipping.")
        return path
    positions = select(len(data), shard, limit)
    classifier = classifier_for(cfg)
    ids, predictions = [], []
    start = time.perf_counter()
    for begin in range(0, len(positions), batch_size):
        samples = [s for s in (data[i] for i in positions[begin:begin + batch_size]) if _scored(s)]
        if not samples:
            continue
        predictions.extend(classifier.predict(samples))
        ids.extend(s.id for s in samples)
        print(f"  {len(ids)}/{len(positions)} ({(time.perf_counter() - start) / len(ids):.2f} s/sample)",
              flush=True)
    store.write(index, count, ids, predictions)
    return path


def calibrate(cfg, split="dev", limit=400):
    from .classifiers.calibration import fit_temperature

    data = dataset_for(cfg, split)
    store = store_for(cfg, split)
    ids = [i for i in select(len(data), limit=limit) if data.ids[i] in store]

    def pairs():
        for i in ids:
            sample = data[i]
            prediction = store.load(sample.id)
            target = sample.grid_target.copy()
            if prediction.instances is not None:   # PanNuke: score detected nuclei only
                target[~prediction.informative] = IGNORE
            yield prediction.probs, target

    best, curve = fit_temperature(pairs())
    store.save_temperature(best)
    return best, curve


def temperature_for(cfg, split):
    value = cfg.classifier.get("temperature", 1.0)
    if value == "auto":
        return store_for(cfg, "dev").temperature()
    return float(value)


def problem_for(cfg, sample, prediction, temperature):
    from dataclasses import replace

    problem_cfg = replace(cfg.problem, temperature=temperature)
    return build_problem(
        prediction.probs, problem_cfg, image=sample.grid_image,
        instances=prediction.instances, informative=prediction.informative,
    )


def _score(sample, prediction, labels):
    target = sample.grid_target
    out = {
        "confusion": pixel_confusion(labels, target),
        "rules": pair_counts(labels),
    }
    if prediction.instances is not None:
        truth = sample.grid_truth_instances
        nuclei = nucleus_counts(
            prediction.instances, instance_classes(labels, prediction.instances),
            truth, instance_classes(target, truth),
        )
        out["nucleus_confusion"] = nuclei["confusion"]
        out["pq"] = nuclei["pq"]
    return out


def solve(cfg, split, shard=(0, 1), run_dir=None, limit=0, save_labels=True):
    data = dataset_for(cfg, split)
    store = store_for(cfg, split)
    temperature = temperature_for(cfg, split)
    solver = get_solver(cfg.solver)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    index, count = shard
    out_path = run_dir / f"shard_{index:04d}_of_{count:04d}.npz"
    if out_path.exists():
        print(f"{out_path} exists; skipping.")
        return out_path

    positions = [i for i in select(len(data), shard, limit) if data.ids[i] in store]
    rows, arrays = [], {"ids": [], "argmax": [], "wcsp": []}
    scores = {"argmax": [], "wcsp": [], "truth_rules": []}
    for n, i in enumerate(positions):
        sample = data[i]
        if not _scored(sample):
            continue
        prediction = store.load(sample.id)
        problem = problem_for(cfg, sample, prediction, temperature)
        init = ArgmaxSolver().solve(problem, np.zeros(problem.shape, dtype=np.int64))
        result = solver.solve(problem, init.labels)
        scores["argmax"].append(_score(sample, prediction, init.labels))
        scores["wcsp"].append(_score(sample, prediction, result.labels))
        scores["truth_rules"].append(pair_counts(sample.grid_target))
        rows.append({
            "id": sample.id, **sample.meta,
            "energy_argmax": init.energy, "energy_wcsp": result.energy,
            "lower_bound": result.lower_bound,
            "changed": float((init.labels != result.labels).mean()),
            "seconds": result.seconds,
            "rounds": max((t.get("round") or 0) for t in result.trace) if result.trace else 0,
        })
        arrays["ids"].append(sample.id)
        arrays["argmax"].append(init.labels.astype(np.uint8))
        arrays["wcsp"].append(result.labels.astype(np.uint8))
        print(f"  {n + 1}/{len(positions)} {sample.id}: E {init.energy:.1f} -> {result.energy:.1f}, "
              f"changed {rows[-1]['changed']:.1%}, {result.seconds:.1f}s", flush=True)

    payload = {"ids": np.array(arrays["ids"])}
    if save_labels:
        payload["argmax"] = np.stack(arrays["argmax"]) if arrays["argmax"] else np.empty(0)
        payload["wcsp"] = np.stack(arrays["wcsp"]) if arrays["wcsp"] else np.empty(0)
    for name in ("argmax", "wcsp"):
        for key in ("confusion", "rules", "nucleus_confusion", "pq"):
            values = [s[key] for s in scores[name] if key in s]
            if values:
                payload[f"{name}_{key}"] = np.stack(values)
    if scores["truth_rules"]:
        payload["truth_rules"] = np.stack(scores["truth_rules"])
    pd.DataFrame(rows).to_csv(run_dir / f"shard_{index:04d}_of_{count:04d}.csv", index=False)
    np.savez_compressed(out_path, **payload)
    return out_path


def summarize(run_dir):
    run_dir = Path(run_dir)
    shards = sorted(run_dir.glob("shard_*_of_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard results in {run_dir}")
    loaded = [dict(np.load(p)) for p in shards]

    def total(key):
        parts = [z[key] for z in loaded if key in z and len(z[key])]
        return np.concatenate(parts).sum(axis=0) if parts else None

    table = []
    for name in ("argmax", "wcsp"):
        row = {"method": name}
        row.update(pixel_report(total(f"{name}_confusion")))
        row.update(rule_report(total(f"{name}_rules")))
        nucleus = total(f"{name}_nucleus_confusion")
        if nucleus is not None:
            row.update(nucleus_report(nucleus, total(f"{name}_pq")))
        table.append(row)
    truth = total("truth_rules")
    if truth is not None:
        table.append({"method": "ground truth", **rule_report(truth)})
    summary = pd.DataFrame(table)
    per_sample = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in shards], ignore_index=True)
    summary["samples"] = len(per_sample)
    summary.loc[summary.method == "wcsp", "mean_seconds"] = per_sample["seconds"].mean()
    summary.loc[summary.method == "wcsp", "mean_changed"] = per_sample["changed"].mean()
    summary.to_csv(run_dir / "summary.csv", index=False)
    per_sample.to_csv(run_dir / "per_sample.csv", index=False)

    if "roi" in per_sample:
        stack = lambda key: np.concatenate([z[key] for z in loaded if key in z])  # noqa: E731
        test = paired_bootstrap(stack("argmax_confusion"), stack("wcsp_confusion"), per_sample["roi"])
        (run_dir / "bootstrap.json").write_text(json.dumps(test, indent=2))
    return summary


def _run_confusions(run_dir):
    """{sample id: {"argmax": (K, K), "wcsp": (K, K)}} and {sample id: roi} of one run."""
    run_dir = Path(run_dir)
    shards = sorted(run_dir.glob("shard_*_of_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard results in {run_dir}")
    out = {}
    for path in shards:
        z = np.load(path)
        for row, sample_id in enumerate(z["ids"].tolist()):
            out[sample_id] = {m: z[f"{m}_confusion"][row] for m in ("argmax", "wcsp")}
    rois = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in shards]).set_index("id")["roi"]
    return out, rois.to_dict()


def compare(run_a, run_b):
    """
    Run b against run a on their shared samples, resampling ROIs:
    argmax b - argmax a, wcsp b - wcsp a, and the difference in WCSP gain
    (wcsp b - argmax b) - (wcsp a - argmax a).
    """
    conf_a, rois = _run_confusions(run_a)
    conf_b, _ = _run_confusions(run_b)
    ids = sorted(set(conf_a) & set(conf_b))
    if not ids:
        raise ValueError(f"{run_a} and {run_b} share no samples.")
    confs = {f"{m}_{tag}": np.stack([conf[i][m] for i in ids])
             for tag, conf in (("a", conf_a), ("b", conf_b)) for m in ("argmax", "wcsp")}
    groups = [rois[i] for i in ids]
    contrasts = {
        "argmax": {"argmax_b": 1, "argmax_a": -1},
        "wcsp": {"wcsp_b": 1, "wcsp_a": -1},
        "wcsp_gain": {"wcsp_b": 1, "argmax_b": -1, "wcsp_a": -1, "argmax_a": 1},
    }
    return {"run_a": str(run_a), "run_b": str(run_b), "samples": len(ids),
            **{name: contrast_bootstrap(confs, groups, w) for name, w in contrasts.items()}}
