"""Breast-only pilot with frozen type-set selection and a gated family expansion."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(EXPERIMENTS / "instances_20261114"))
sys.path.append(str(EXPERIMENTS / "presence_20261114"))
import prepare as inference

presence = None
CACHE = HERE / "cache"
SEED = 20261122
HASH = "789b2be8bdaf532627f584a009f83b3ba06e04c0e55c1635252ff12785ef57e8"


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    target = HERE / name
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)


def configure():
    inference.HERE = HERE
    inference.CACHE = CACHE
    if presence is not None:
        presence.nucleus.CACHE = CACHE
        presence.source.HERE = HERE
    split = read(HERE / "split.json")
    assert len(split["test"]) >= 60, "Stop: fewer than 60 test patches."
    original_read = inference.load

    def load_split(path):
        if path == HERE / "split.json":
            return dict(split, train=split["fit"])
        return original_read(path)

    inference.load = load_split
    return split


def prepare():
    split = configure()
    assert not set(row["image_index"] for row in split["test"]) & set(split["excluded_ids"])
    checkpoint = REPO / "benchmarks/pannuke/data/cache/models/hovernet_fast-pannuke.pth"
    digest = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    assert digest == HASH, (digest, HASH)
    write("protocol.json", dict(
        stage="Pilot type-set WCSP, gated expansion per user's last instruction.",
        seed=SEED, checkpoint_sha256=digest,
        selection="Highest pooled five-class dev mIoU. Within 0.0001 prefer smaller strength, plain presence, then vote/mean/log_mean.",
        pilot_gate="Expand if the tuned type-set winner has positive 95% paired intervals vs Raw TP and vote, and a positive 95% paired interval vs the frozen global type-set model. A fixed-family Holm adjustment is reported when all eight families exist.",
        exposure="Test GT extraction supports cache preparation only. Test predictions and metrics wait for frozen.json.",
        environment="Reuse existing Python 3.12 virtual environment and installed dependencies. uv executable unavailable.",
        sources={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (EXPERIMENTS / "presence_20261114/run.py",
                              EXPERIMENTS / "instances_20261114/prepare.py",
                              EXPERIMENTS / "instances_20261114/run.py")},
    ))
    inference.prepare()
    print("Preparation complete", flush=True)


def merged_priors(rows):
    boundaries = [8, 20, 40]
    observed = []
    for row in rows:
        instance_mask, labels = presence.nucleus.load_gt(row["image_index"])
        types = labels[1:] - 1
        mask = int(np.bitwise_or.reduce(1 << types.astype(int), initial=0))
        observed.append((len(types), mask))
    while boundaries:
        bins = np.bincount([np.searchsorted(boundaries, n, side="right") for n, mask in observed],
                          minlength=len(boundaries) + 1)
        rare = np.flatnonzero(bins < 10)
        if not len(rare):
            break
        index = int(rare[0])
        if index == len(boundaries):
            boundaries.pop()
        else:
            boundaries.pop(index)
    counts = np.zeros((len(boundaries) + 1, 32), int)
    for n, mask in observed:
        counts[np.searchsorted(boundaries, n, side="right"), mask] += 1
    pooled = (counts.sum(0) + 1) / (counts.sum() + 32)
    band = (counts + 16 * pooled) / (counts.sum(1, keepdims=True) + 16)
    write("priors_and_tables.json", dict(
        fit_patch_ids=[r["image_index"] for r in rows], type_set=dict(
            boundaries=boundaries, counts=counts.tolist(), band_probability=band.tolist(),
            global_probability=pooled.tolist(), minimum_band_support=10)))
    return boundaries, band


def dev():
    split = configure()
    assert not (HERE / "frozen.json").exists()
    boundaries, band = merged_priors(split["fit"])
    presence.BANDS = boundaries
    data = presence.records(split["dev"], band, np.ones(5) / 5)
    candidates = [
        presence.config(f"{kind}_{mode}_{strength:g}", mode, kind, strength)
        for kind in ("vote", "mean", "log_mean")
        for mode in ("presence", "presence_evidence")
        for strength in (.25, .5, 1, 2, 4)
    ]
    rows = []
    for config in candidates:
        result = presence.evaluate(config, data)[0]
        rows.append(result)
        print(config["name"], result["bio_miou"], flush=True)
    best = max(r["bio_miou"] for r in rows)
    close = [r for r in rows if r["bio_miou"] >= best - .0001]
    selected = min(close, key=lambda r: (
        r["config"]["strength"], r["config"]["mode"] != "presence",
        ("vote", "mean", "log_mean").index(r["config"]["unary"])))
    write("development.json", dict(type_set=rows))
    write("frozen.json", dict(type_set=selected["config"], boundaries=boundaries,
                             best_dev_bio_miou=best, chosen_dev_bio_miou=selected["bio_miou"],
                             test_used=False))
    print("FROZEN", selected, flush=True)


def resample_score(matrix, weights):
    counts = (weights @ matrix.reshape(len(matrix), -1)).reshape(-1, 6, 6)
    return presence.nucleus.iou(counts)[:, 1:]


def test():
    split = configure()
    frozen = read(HERE / "frozen.json")
    assert not (HERE / "evaluation.json").exists(), "Completed test is immutable."
    tables = read(HERE / "priors_and_tables.json")["type_set"]
    presence.BANDS = frozen["boundaries"]
    data = presence.records(split["test"], np.array(tables["band_probability"]), np.ones(5) / 5)
    reference_cache = CACHE / "global_reference"
    reference_root = reference_cache if reference_cache.exists() else EXPERIMENTS / "presence_20261114"
    global_config = read(reference_root / "frozen.json")["presence"]
    global_tables = read(reference_root / "priors.json")
    write("global_reference.json", dict(
        config=global_config, source_commit="19e8df5",
        sha256={name: hashlib.sha256((reference_root / name).read_bytes()).hexdigest()
                for name in ("frozen.json", "priors.json")},
    ))
    presence.BANDS = [8, 20, 40]
    global_data = presence.records(split["test"], np.array(global_tables["band_probability"]),
                                   np.array(global_tables["type_frequency"]))
    unary_control = presence.config("matched_unary", "unary", frozen["type_set"]["unary"])
    choices = [("vote", presence.config("vote", "instance_vote"), data),
               ("matched_unary", unary_control, data),
               ("breast_type_set", frozen["type_set"], data),
               ("global_type_set", global_config, global_data)]
    matrices = {"raw_tp": np.stack([presence.nucleus.confusion(p["gt"], p["raw"]) for p in data])}
    results = {}
    started = time.perf_counter()
    for name, config, patches in choices:
        result, matrices[name], overlap, denominator = presence.evaluate(config, patches)
        results[name] = result
    n = len(data)
    draws = np.random.default_rng(SEED).integers(n, size=(5000, n))
    weights = np.zeros((5000, n), int)
    np.add.at(weights, (np.arange(5000)[:, None], draws), 1)
    bootstrap = {name: resample_score(mat, weights) for name, mat in matrices.items()}
    comparisons = {}
    for control in ("raw_tp", "vote", "global_type_set", "matched_unary"):
        delta = bootstrap["breast_type_set"] - bootstrap[control]
        point = presence.nucleus.iou(matrices["breast_type_set"].sum(0)) - presence.nucleus.iou(matrices[control].sum(0))
        patch_delta = presence.nucleus.score(matrices["breast_type_set"]) - presence.nucleus.score(matrices[control])
        comparisons[control] = dict(delta=float(point[1:].mean()),
            ci95=np.percentile(delta.mean(1), [2.5, 97.5]).tolist(),
            reserved_eight_family_ci=np.percentile(delta.mean(1), [.3125, 99.6875]).tolist(),
            no_dead_delta=float(point[[1, 2, 3, 5]].mean()),
            no_dead_ci95=np.percentile(delta[:, [0, 1, 2, 4]].mean(1), [2.5, 97.5]).tolist(),
            wlt=[int((patch_delta > 1e-12).sum()), int((patch_delta < -1e-12).sum()),
                 int((abs(patch_delta) <= 1e-12).sum())])
    pilot_passed = (comparisons["raw_tp"]["ci95"][0] > 0
                    and comparisons["vote"]["ci95"][0] > 0
                    and comparisons["global_type_set"]["ci95"][0] > 0)
    np.savez_compressed(HERE / "confusions.npz", **matrices)
    with (HERE / "per_patch.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_index", *matrices])
        for i, patch in enumerate(data):
            writer.writerow([patch["index"], *[float(presence.nucleus.score(mat[i])) for mat in matrices.values()]])
    support = np.array([np.bincount(p["truth"], minlength=5) for p in data])
    for name, result in results.items():
        result["without_dead"] = float(np.array(result["per_class_iou"])[[0, 1, 2, 4]].mean())
    write("evaluation.json", dict(stage="pilot", n_test=n, results=results,
        raw_miou=float(presence.nucleus.score(matrices["raw_tp"].sum(0))),
        comparisons=comparisons, pilot_passed=bool(pilot_passed),
        truth_counts=support.sum(0).tolist(), patch_support=(support > 0).sum(0).tolist(),
        seconds_evaluation=time.perf_counter() - started,
        inference_sha256=read(HERE / "inference_test.json")["checkpoint_sha256"]))
    print(json.dumps(dict(comparisons=comparisons, pilot_passed=bool(pilot_passed)), indent=2), flush=True)


def infer_group(group, use_gpu):
    # The historical dev entry point checks identical IDs against its own
    # cache. New Breast dev IDs use the same inference path with a remapped
    # record list, avoiding that historical reference-only assertion.
    original_load = inference.load
    original_write = inference.write

    def group_records(path):
        value = original_load(path)
        if path == HERE / "split.json":
            return dict(value, test=value[group])
        return value

    def group_receipt(name, value):
        if name == "inference_test.json":
            name = f"inference_{group}.json"
        original_write(name, value)

    inference.load = group_records
    inference.write = group_receipt
    if use_gpu:
        gpu_inference("test")
    else:
        inference.infer("test")


def gpu_inference(group):
    original = inference.prior_module

    class DeviceModel:
        def __init__(self, model):
            self.model = model.to("mps")

        def __call__(self, tensor):
            output = self.model(tensor.to("mps"))
            return {key: value.cpu() for key, value in output.items()}

    def previous_module():
        module = original()
        original_loader = module.load_model

        def model_loader():
            model, hashes = original_loader()
            return DeviceModel(model), hashes

        module.load_model = model_loader
        return module

    inference.prior_module = previous_module
    inference.infer(group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer_dev", "dev", "infer_test", "test", "all_dev", "all_test"])
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.stage in ("dev", "test", "all_dev", "all_test"):
        # The objective imports four context names from its preparation
        # script. Supply those names without running its old split logic.
        helper = types.ModuleType("prepare_fresh")
        helper.HERE = EXPERIMENTS / "presence_20261114"
        helper.CACHE = helper.HERE / "cache"
        helper.PREVIOUS = EXPERIMENTS / "instances_20261114"
        helper.source = inference
        sys.modules["prepare_fresh"] = helper
        spec = importlib.util.spec_from_file_location(
            "breast_presence_objective", EXPERIMENTS / "presence_20261114/run.py"
        )
        presence = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(presence)
    if args.stage == "prepare":
        prepare()
    elif args.stage in ("infer_dev", "infer_test"):
        configure()
        group = args.stage.removeprefix("infer_")
        infer_group(group, args.gpu)
    elif args.stage == "dev":
        dev()
    elif args.stage == "test":
        test()
    else:
        import families
        configure()
        if args.stage == "all_dev":
            families.development(presence)
        else:
            families.testing(presence)
