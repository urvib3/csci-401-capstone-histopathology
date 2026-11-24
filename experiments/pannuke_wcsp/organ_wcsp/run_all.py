"""Run every PanNuke WCSP family on one organ: fit, tune on dev, freeze, test once.

Usage (from the repository root):
    .venv/bin/python experiments/pannuke_wcsp/organ_wcsp/run_all.py prepare --organ Adrenal_gland --folds 1,2,3
    .venv/bin/python experiments/pannuke_wcsp/organ_wcsp/run_all.py infer   --organ Adrenal_gland
    .venv/bin/python experiments/pannuke_wcsp/organ_wcsp/run_all.py dev     --organ Adrenal_gland
    .venv/bin/python experiments/pannuke_wcsp/organ_wcsp/run_all.py test    --organ Adrenal_gland

The objectives and solvers come from the earlier experiment folders, which are
imported and never modified. Only the patches, the fitted tables and the tuned
settings change: everything is fit on this organ's fit split and tuned on its
dev split. The test split is scored once, after frozen.json is written.
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
CACHE = HERE / "cache"
DATA_DIR = HERE.parents[2] / "benchmarks" / "pannuke" / "data"
FOLD_OFFSETS = {1: 10000, 2: 0, 3: 30000}
SEED = 20261122
BOOTSTRAP_REPLICATES = 5000
TIE_THRESHOLD = 0.0001
NO_DEAD_CLASSES = [1, 2, 3, 5]
CLASS_NAMES = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Non-neoplastic epithelial"]

INSTANCES_DIR = EXPERIMENTS / "instances_20261114"
PRESENCE_DIR = EXPERIMENTS / "presence_20261114"
PRESENCE_SCALE_DIR = EXPERIMENTS / "presence_scale_20261114"
HIERARCHY_DIR = EXPERIMENTS / "hierarchy_20261109"
SCALING_DIR = EXPERIMENTS / "scaling_siblings_20261109"
FAMILY_DECAY_DIR = EXPERIMENTS / "family_decay_20261117"
D5_DIR = EXPERIMENTS / "pannuke_d5"

FAMILIES = ["pixel_table", "hierarchy", "scaling_siblings", "decay",
            "nucleus_pairwise", "type_counts", "type_set", "raw_component"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Numba's on-disk cache looks the module up by name when it reloads a function.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The presence experiment imports the instance experiment ("run" and "prepare")
# by module name, so its folder must be importable before it is loaded.
sys.path.insert(0, str(PRESENCE_DIR))
presence = load_module("presence_run", PRESENCE_DIR / "run.py")
nucleus = presence.nucleus
source = presence.source
raw_star = load_module("raw_star_run", PRESENCE_DIR / "raw_star.py")
hierarchy_core = load_module("hierarchy_core", HIERARCHY_DIR / "core.py")
pixel_core = load_module("pixel_core", D5_DIR / "core.py")

nucleus.CACHE = CACHE
presence.CACHE = CACHE
raw_star.CACHE = CACHE
source.CACHE = CACHE

GLOBAL_BANDS = [8, 20, 40]
HIERARCHY_SIZES = np.array(hierarchy_core.SIZES, dtype=float)
POTTS_6 = 1 - np.eye(6)


# ---------------------------------------------------------------- small helpers

def read_json(path):
    return json.loads(Path(path).read_text())


def to_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=to_plain) + "\n")
    temporary.replace(path)


def output_dir(organ):
    return HERE / organ.lower()


def class_iou(matrix):
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    union = matrix.sum(axis=-1) + matrix.sum(axis=-2) - diagonal
    return np.divide(diagonal, union, out=np.zeros_like(diagonal, dtype=float), where=union > 0)


def biological_miou(matrix):
    return float(class_iou(matrix)[1:].mean())


def confusion(gt, prediction):
    return np.bincount((gt.astype(np.int64) * 6 + prediction).ravel(), minlength=36).reshape(6, 6)


# ---------------------------------------------------------------- split and data

def previously_used_ids():
    """Fold 2 patch IDs used by any earlier experiment. Folds 1 and 3 were never used."""
    split = read_json(PRESENCE_SCALE_DIR / "split.json")
    used = set(split["excluded_ids"])
    for row in split["test"]:
        used.add(row["image_index"])
    return used


def patch_id(fold, index_in_fold):
    """Fold 2 keeps its original index, so earlier exclusion lists still apply."""
    return FOLD_OFFSETS[fold] + index_in_fold


def fold_archive(fold):
    return DATA_DIR / f"fold_{fold}.zip"


def open_fold_member(fold, suffix):
    archive = zipfile.ZipFile(fold_archive(fold))
    names = []
    for name in archive.namelist():
        if name.endswith("/" + suffix):
            names.append(name)
    assert len(names) == 1, names
    return archive, names[0]


def fold_tissues(fold):
    archive, member = open_fold_member(fold, "types.npy")
    with archive, archive.open(member) as file:
        return np.load(file)


def stream_fold_arrays(fold, array_name, wanted_indices):
    """Read selected patches from a large .npy inside the zip without extracting it."""
    archive, member = open_fold_member(fold, array_name)
    wanted = set(wanted_indices)
    with archive, archive.open(member) as file:
        version = np.lib.format.read_magic(file)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
        assert not fortran_order and not dtype.hasobject
        bytes_per_patch = int(np.prod(shape[1:])) * dtype.itemsize
        for index in range(max(wanted) + 1):
            raw_bytes = file.read(bytes_per_patch)
            assert len(raw_bytes) == bytes_per_patch
            if index in wanted:
                yield index, np.frombuffer(raw_bytes, dtype=dtype).reshape(shape[1:])


def make_split(organ, out, folds):
    used_fold_2 = previously_used_ids()
    # Fold 2 goes first, so a fold-2-only split is the plain protocol:
    # one permutation from default_rng(SEED) over the unused organ patches.
    ordered_folds = sorted(folds, key=lambda fold: (fold != 2, fold))
    rng = np.random.default_rng(SEED)
    fit, dev, test = [], [], []
    fold_summary = {}
    for fold in ordered_folds:
        tissues = fold_tissues(fold)
        available_organs = sorted(set(str(t) for t in tissues))
        if organ not in available_organs:
            raise SystemExit(f"Unknown organ {organ!r}. Choose one of {available_organs}")
        eligible = []
        total_in_fold = 0
        for index, tissue in enumerate(tissues):
            if str(tissue) != organ:
                continue
            total_in_fold += 1
            if fold == 2 and index in used_fold_2:
                continue
            eligible.append(index)

        shuffled = rng.permutation(eligible).tolist()
        fit_count = int(round(0.4 * len(shuffled)))
        dev_count = int(round(0.2 * len(shuffled)))
        groups = dict(fit=shuffled[:fit_count], dev=shuffled[fit_count:fit_count + dev_count],
                      test=shuffled[fit_count + dev_count:])
        for group_name, target in (("fit", fit), ("dev", dev), ("test", test)):
            for index in sorted(groups[group_name]):
                target.append(dict(image_index=patch_id(fold, int(index)), fold=fold,
                                   index_in_fold=int(index), tissue=organ))
        fold_summary[fold] = dict(organ_patches=total_in_fold, previously_used_removed=total_in_fold - len(eligible),
                                  fit=len(groups["fit"]), dev=len(groups["dev"]), test=len(groups["test"]))

    split = dict(
        organ=organ, folds=ordered_folds, seed=SEED, fractions=dict(fit=0.4, dev=0.2, test=0.4),
        per_fold=fold_summary, fit=fit, dev=dev, test=test, scored=dev + test,
        patch_id_rule="image_index = fold offset + index within fold; offsets " + json.dumps(FOLD_OFFSETS),
        excluded_source="Fold 2 only: excluded_ids plus every test ID in presence_scale_20261114/split.json",
        stratification="Each fold is shuffled and split 40/20/40 separately, then pooled.",
        independence="Patch-disjoint only. PanNuke has no patient or slide IDs. The folds HoVer-Net was trained on are unknown.",
    )
    write_json(out / "split.json", split)
    return split


def prepare(organ, folds):
    out = output_dir(organ)
    out.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    if (out / "split.json").exists():
        split = read_json(out / "split.json")
        print("Using the existing split.json", flush=True)
    else:
        split = make_split(organ, out, folds)
    for fold, summary in split["per_fold"].items():
        print(f"{organ} fold {fold}: {summary}", flush=True)
    print(f'Total: fit {len(split["fit"])} / dev {len(split["dev"])} / test {len(split["test"])}', flush=True)
    if len(split["test"]) < 60:
        print("WARNING: fewer than 60 test patches. Tell the team before continuing.", flush=True)

    for fold in split["folds"]:
        fold_rows = [r for r in split["fit"] + split["scored"] if r["fold"] == fold]
        missing_gt = [r for r in fold_rows if not (CACHE / f'gt_{r["image_index"]:04d}.npz').exists()]
        if missing_gt:
            print(f"Fold {fold}: extracting ground truth for {len(missing_gt)} patches", flush=True)
            id_by_index = {r["index_in_fold"]: r["image_index"] for r in missing_gt}
            for index, masks in stream_fold_arrays(fold, "masks.npy", id_by_index):
                instances, types = source.ground_truth(masks)
                np.savez_compressed(CACHE / f"gt_{id_by_index[index]:04d}.npz", instances=instances, types=types)

        scored_rows = [r for r in split["scored"] if r["fold"] == fold]
        missing_rgb = [r for r in scored_rows if not (CACHE / f'rgb_{r["image_index"]:04d}.npy').exists()]
        if missing_rgb:
            print(f"Fold {fold}: extracting RGB for {len(missing_rgb)} patches", flush=True)
            id_by_index = {r["index_in_fold"]: r["image_index"] for r in missing_rgb}
            for index, image in stream_fold_arrays(fold, "images.npy", id_by_index):
                converted = image.astype(np.uint8)
                assert np.array_equal(image, converted)
                np.save(CACHE / f"rgb_{id_by_index[index]:04d}.npy", converted)
    print("Prepared", flush=True)


def infer(organ, folds=None):
    """Same frozen checkpoint and full-patch tiling as instances_20261114/prepare.py."""
    source.HERE = output_dir(organ)
    source.infer("scored")


def semantic_gt(index):
    with np.load(CACHE / f"gt_{index:04d}.npz") as file:
        return file["types"][file["instances"]]


# ---------------------------------------------------------------- fitting on the fit split

def fit_type_set_prior(fit_records):
    """Patch type-set prior from presence_20261114, with sparse nucleus-count bands merged."""
    label_lists = []
    for row in fit_records:
        _, types = nucleus.load_gt(row["image_index"])
        label_lists.append(types[1:].astype(int) - 1)

    bands = list(GLOBAL_BANDS)
    while bands:
        patches_per_band = np.zeros(len(bands) + 1, int)
        for labels in label_lists:
            patches_per_band[np.searchsorted(bands, len(labels), side="right")] += 1
        smallest = int(patches_per_band.argmin())
        if patches_per_band[smallest] >= 10:
            break
        # Merge the sparse band into a neighbour by deleting the cut between them.
        if smallest == 0:
            del bands[0]
        else:
            del bands[smallest - 1]

    set_count = 1 << presence.N
    counts = np.zeros((len(bands) + 1, set_count), int)
    type_totals = np.ones(presence.N)
    for labels in label_lists:
        np.add.at(type_totals, labels, 1)
        type_mask = 0
        for label in labels:
            type_mask = type_mask | (1 << int(label))
        counts[np.searchsorted(bands, len(labels), side="right"), type_mask] += 1
    global_probability = (counts.sum(0) + 1) / (counts.sum() + set_count)
    band_probability = (counts + 16 * global_probability) / (counts.sum(1, keepdims=True) + 16)
    type_frequency = type_totals / type_totals.sum()
    return dict(bands=bands, set_counts=counts, band_probability=band_probability,
                type_frequency=type_frequency)


def fit_hierarchy_tables(fit_records):
    tables = []
    for size in hierarchy_core.SIZES:
        pair_counts = np.zeros((6, 6), np.int64)
        for row in fit_records:
            gt = semantic_gt(row["image_index"])
            pair_counts += hierarchy_core.counts(hierarchy_core.majority(gt, size))
        table, _ = hierarchy_core.fit_table(pair_counts)
        tables.append(table)
    return np.array(tables)


def fit_pixel_table(fit_records):
    pair_counts = np.zeros((6, 6), np.int64)
    for row in fit_records:
        pair_counts += pixel_core.adjacency(semantic_gt(row["image_index"]))
    return pixel_core.fit(pair_counts)["final"]


def fit_nucleus_tables(fit_records):
    # nucleus.fit writes tables.json through its own module; keep that out of other folders.
    original_write = nucleus.write
    nucleus.write = lambda name, value: None
    try:
        tables = nucleus.fit(fit_records)
    finally:
        nucleus.write = original_write
    return tables


def fit_everything(split):
    return dict(
        type_set=fit_type_set_prior(split["fit"]),
        hierarchy=fit_hierarchy_tables(split["fit"]),
        pixel=fit_pixel_table(split["fit"]),
        nucleus=fit_nucleus_tables(split["fit"]),
    )


def global_tables():
    """Tables and priors exactly as frozen by the original (all-organ) experiments."""
    priors = read_json(PRESENCE_DIR / "priors.json")
    nucleus_json = read_json(INSTANCES_DIR / "tables.json")
    return dict(
        type_set=dict(bands=list(GLOBAL_BANDS), band_probability=np.array(priors["band_probability"]),
                      type_frequency=np.array(priors["type_frequency"])),
        hierarchy=np.array(read_json(HIERARCHY_DIR / "tables.json")["final"]),
        pixel=np.array(read_json(D5_DIR / "tables.json")["final"]),
        nucleus={name: np.array(nucleus_json[name]) for name in ("potts", "hybrid", "area")},
    )


def save_tables(out, organ_tables):
    write_json(out / "priors_and_tables.json", dict(
        type_set_prior=organ_tables["type_set"],
        hierarchy_tables=organ_tables["hierarchy"],
        pixel_table=organ_tables["pixel"],
        nucleus_tables=organ_tables["nucleus"],
        note="All fitted on this organ's fit split ground truth only.",
    ))


# ---------------------------------------------------------------- patch loading

def load_patches(records, organ_tables):
    """Load probabilities, masks, ground truth and the nucleus-level inputs for each patch."""
    type_set = organ_tables["type_set"]
    presence.BANDS = type_set["bands"]
    nucleus_records = presence.records(records, type_set["band_probability"], type_set["type_frequency"])
    component_inputs = raw_star.inputs(records)

    patches = []
    for row, nucleus_patch, component in zip(records, nucleus_records, component_inputs):
        index = row["image_index"]
        with np.load(CACHE / f"pred_{index:04d}.npz") as file:
            probabilities = file["p"].astype(float)
        nucleus_patch["probabilities"] = probabilities
        nucleus_patch["component"] = component
        nucleus_patch["block_margins"] = block_margins(probabilities)
        patches.append(nucleus_patch)
    return patches


def block_margins(probabilities):
    """Top-two margin per block, for the confidence-weighted fidelity schedule."""
    margins = []
    for block in hierarchy_core.block_probabilities(probabilities)[1:]:
        top_two = np.partition(block, -2, axis=-1)[..., -2:].reshape(-1, 2)
        margins.append(0.25 + 0.75 * (top_two[:, 1] - top_two[:, 0]))
    return margins


def set_probability_for(patch, type_set):
    nuclei = len(patch["baseline"])
    band = np.searchsorted(type_set["bands"], nuclei, side="right")
    return type_set["band_probability"][band, presence.MASKS]


# ---------------------------------------------------------------- methods

def method(family, name, kind, tables="organ", **params):
    return dict(family=family, name=name, kind=kind, tables=tables, params=params)


def hierarchy_settings(beta=None, parent=None, gamma=None, adaptive=False, table="hybrid", scaled=False):
    beta = [0.3] * 6 if beta is None else list(beta)
    parent = [0.1] * 5 if parent is None else list(parent)
    gamma = [0.0] * 5 if gamma is None else list(gamma)
    if scaled:
        # scaling_siblings_20261109: sibling x side, parent-child x child area, block unary x area.
        beta = [b * HIERARCHY_SIZES[level] for level, b in enumerate(beta)]
        parent = [p * HIERARCHY_SIZES[level] ** 2 for level, p in enumerate(parent)]
        gamma = [g * HIERARCHY_SIZES[level + 1] ** 2 for level, g in enumerate(gamma)]
    return dict(beta=beta, parent=parent, gamma=gamma, adaptive=adaptive, table=table)


def solve_hierarchy(patch, settings, hybrid_tables):
    """The family_decay_20261117 solver: per-level sibling, parent and block-unary weights."""
    graph = HIERARCHY_GRAPH
    labels, unary = hierarchy_core.initial_unary(patch["probabilities"], 1.0)
    levels = list(zip(graph["offsets"][:-1], graph["offsets"][1:]))
    for level_number, (start, end) in enumerate(levels[1:]):
        weight = settings["gamma"][level_number]
        if settings["adaptive"]:
            unary[start:end] *= weight * patch["block_margins"][level_number][:, None]
        else:
            unary[start:end] *= weight

    if settings["table"] == "hybrid":
        sibling_tables = hybrid_tables
    elif settings["table"] == "potts":
        sibling_tables = np.repeat(POTTS_6[None], 6, axis=0)
    else:
        sibling_tables = np.zeros((6, 6, 6))
    sibling = sibling_tables * np.array(settings["beta"])[:, None, None] / 4
    parent = POTTS_6[None] * np.array(settings["parent"])[:, None, None]
    costs = np.concatenate((sibling, parent))

    previous_energy = hierarchy_core.energy(labels, unary, costs, graph)
    for _ in range(100):
        changes = 0
        for reverse in (False, True):
            changes += hierarchy_core.sweep(labels, unary, costs, graph["ptr"], graph["dst"], graph["typ"], reverse)
        current_energy = hierarchy_core.energy(labels, unary, costs, graph)
        assert current_energy <= previous_energy + 1e-7
        previous_energy = current_energy
        if changes == 0:
            break
    return labels[:65536].reshape(256, 256).astype(np.uint8)


def nucleus_labels_to_image(patch, labels):
    image = np.r_[0, labels + 1][patch["instances"]]
    assert np.array_equal(image > 0, patch["instances"] > 0)
    return image


def predict(item, patch, tables_by_source):
    tables = tables_by_source[item["tables"]]
    kind = item["kind"]
    params = item["params"]

    if kind == "raw":
        return patch["raw"]
    if kind == "vote":
        return nucleus_labels_to_image(patch, patch["baseline"])
    if kind == "pixel_table":
        labels, _ = pixel_core.solve(patch["probabilities"], tables["pixel"], params["beta"])
        return labels
    if kind == "hierarchy":
        return solve_hierarchy(patch, params, tables["hierarchy"])
    if kind == "nucleus_pairwise":
        unary = patch["cost"][params["unary"]]
        if params["area_weight"]:
            bins, valid = patch["area"]
            unary = unary + params["area_weight"] * tables["nucleus"]["area"][bins] * valid[:, None]
        labels = nucleus.solve(unary, patch["graph"], tables["nucleus"][params["table"]],
                               params["beta"], patch["baseline"])
        return nucleus_labels_to_image(patch, labels)
    if kind == "presence":
        # Swap in this table source's type-set prior and fitted type frequencies.
        type_set = tables["type_set"]
        adjusted = dict(patch, set_probability=set_probability_for(patch, type_set))
        nuclei = len(patch["baseline"])
        adjusted["expected_blend"] = 0.8 * patch["expected_model"] + 0.2 * type_set["type_frequency"] * nuclei
        labels = presence.predict(params, adjusted)
        return nucleus_labels_to_image(patch, labels)
    if kind == "component":
        return raw_star.refine(patch["component"], params["strength"], params["limit"], params["adaptive"])
    raise ValueError(kind)


def presence_params(mode, unary, strength):
    return dict(mode=mode, unary=unary, strength=strength)


def development_grid():
    """Every family's candidates, in tie-break order (simplest or weakest first)."""
    grid = {family: [] for family in FAMILIES}

    for beta in (0.05, 0.10, 0.15, 0.30, 0.50):
        grid["pixel_table"].append(method("pixel_table", f"pixel_table_beta{beta:g}", "pixel_table", beta=beta))

    for gamma in (0.0, 0.1, 0.3, 1.0):
        for sibling, parent in ((0.1, 0.03), (0.3, 0.1), (0.6, 0.3)):
            settings = hierarchy_settings(beta=[sibling] * 6, parent=[parent] * 5, gamma=[gamma] * 5)
            name = f"hierarchy_sib{sibling:g}_par{parent:g}_gamma{gamma:g}"
            grid["hierarchy"].append(method("hierarchy", name, "hierarchy", **settings))

    for table in ("zero", "potts", "hybrid"):
        settings = hierarchy_settings(table=table)
        grid["scaling_siblings"].append(method("scaling_siblings", f"siblings_{table}", "hierarchy", **settings))
    for parent in (0.01, 0.03, 0.1, 0.3, 0.5):
        settings = hierarchy_settings(parent=[parent] * 5, gamma=[0.3] * 5, scaled=True)
        grid["scaling_siblings"].append(method("scaling_siblings", f"scaled_parent{parent:g}", "hierarchy", **settings))

    # decay_20261114 schedules: sibling weight shrinks by a factor d per level.
    for d in (0.8, 0.6, 0.4, 0.2):
        schedule = [0.3 * d ** power for power in range(5, -1, -1)]
        grid["decay"].append(method("decay", f"decay_blocks_{d:g}", "hierarchy",
                                    **hierarchy_settings(beta=[0.3] + schedule[1:])))
        grid["decay"].append(method("decay", f"decay_all_{d:g}", "hierarchy",
                                    **hierarchy_settings(beta=schedule)))
        grid["decay"].append(method("decay", f"decay_pixels_{d:g}", "hierarchy",
                                    **hierarchy_settings(beta=schedule[:1] + [0.3] * 5)))
        grid["decay"].append(method("decay", f"decay_potts_blocks_{d:g}", "hierarchy",
                                    **hierarchy_settings(beta=[0.3] + schedule[1:], table="potts")))
    # family_decay_20261117 schedules: parent, sibling and block-unary weights vary with block side.
    for power in (-1, -0.5, 0.5, 1):
        parent_schedule = schedule_by_side(HIERARCHY_SIZES[:5], power, 0.1)
        sibling_schedule = [0.3] + list(schedule_by_side(HIERARCHY_SIZES[1:], power, 0.3))
        unary_schedule = schedule_by_side(HIERARCHY_SIZES[1:], power, 0.3)
        grid["decay"].append(method("decay", f"parent_{power:g}", "hierarchy",
                                    **hierarchy_settings(parent=parent_schedule)))
        grid["decay"].append(method("decay", f"sibling_{power:g}", "hierarchy",
                                    **hierarchy_settings(beta=sibling_schedule)))
        grid["decay"].append(method("decay", f"unary_{power:g}", "hierarchy",
                                    **hierarchy_settings(gamma=unary_schedule)))
    grid["decay"].append(method("decay", "parent_area", "hierarchy",
                                **hierarchy_settings(parent=schedule_by_side(HIERARCHY_SIZES[:5], 2, 0.1))))
    grid["decay"].append(method("decay", "unary_adaptive", "hierarchy",
                                **hierarchy_settings(gamma=[0.3] * 5, adaptive=True)))
    grid["decay"].append(method("decay", "unary_adaptive_fine", "hierarchy",
                                **hierarchy_settings(gamma=schedule_by_side(HIERARCHY_SIZES[1:], -0.5, 0.3), adaptive=True)))
    grid["decay"].append(method("decay", "unary_adaptive_coarse", "hierarchy",
                                **hierarchy_settings(gamma=schedule_by_side(HIERARCHY_SIZES[1:], 0.5, 0.3), adaptive=True)))

    for config in nucleus.configs():
        if config["beta"] > 0:
            params = dict(unary=config["unary"], table=config["table"], beta=config["beta"],
                          area_weight=config["area_weight"])
            grid["nucleus_pairwise"].append(method("nucleus_pairwise", "nucleus_" + config["name"],
                                                   "nucleus_pairwise", **params))
    # Prefer lower beta, then no area term, then Potts (the original tie-break order).
    grid["nucleus_pairwise"].sort(key=lambda item: (item["params"]["beta"], item["params"]["area_weight"],
                                                    item["params"]["table"] != "potts"))

    for strength in (0.25, 0.5, 1.0, 2.0, 4.0):
        for unary in ("vote", "mean", "log_mean"):
            for mode in ("count_model", "count_blend"):
                name = f"{unary}_{mode}_{strength:g}"
                grid["type_counts"].append(method("type_counts", name, "presence",
                                                  **presence_params(mode, unary, strength)))
            for mode in ("presence", "presence_evidence"):
                name = f"{unary}_{mode}_{strength:g}"
                grid["type_set"].append(method("type_set", name, "presence",
                                               **presence_params(mode, unary, strength)))

    for strength in (0.10, 0.25, 0.50, 1.0, 2.0):
        for limit in (128, 256, 65536):
            for adaptive in (False, True):
                name = f"component_s{strength:g}_limit{limit}_{'adaptive' if adaptive else 'flat'}"
                grid["raw_component"].append(method("raw_component", name, "component",
                                                    strength=strength, limit=limit, adaptive=adaptive))
    return grid


def without_name(config):
    params = dict(config)
    params.pop("name", None)
    return params


def schedule_by_side(sides, power, mean):
    raw = np.asarray(sides, dtype=float) ** power
    return list(mean * raw / raw.mean())


def controls():
    """Matched controls. Each family is compared against the method its change is layered on."""
    return dict(
        raw=method("control", "raw_tp", "raw"),
        vote=method("control", "fixed_mask_vote", "vote"),
        v1=method("control", "organ_v1", "hierarchy", **hierarchy_settings()),
        v2=method("control", "organ_v2", "hierarchy", **hierarchy_settings(gamma=[0.3] * 5)),
    )


FAMILY_CONTROL = dict(pixel_table="raw", hierarchy="raw", scaling_siblings="v2", decay="v1",
                      nucleus_pairwise="vote", type_counts="vote", type_set="vote", raw_component="raw")


def global_methods():
    """Each family's original frozen config, with the original all-organ tables."""
    instances_frozen = read_json(INSTANCES_DIR / "frozen.json")["wcsp"]
    presence_frozen = read_json(PRESENCE_DIR / "frozen.json")
    component_frozen = read_json(PRESENCE_DIR / "raw_only_frozen.json")
    hierarchy_frozen = read_json(HIERARCHY_DIR / "frozen_config.json")
    scaling_frozen = read_json(SCALING_DIR / "scaling_frozen.json")
    family_frozen = read_json(FAMILY_DECAY_DIR / "frozen.json")["configurations"]
    combined = next(c for c in family_frozen if c["name"] == "combined")

    return dict(
        pixel_table=method("pixel_table", "global_pixel_table_beta0.3", "pixel_table", "global", beta=0.30),
        hierarchy=method("hierarchy", "global_v1", "hierarchy", "global",
                         **hierarchy_settings(beta=[hierarchy_frozen["sibling"]] * 6,
                                              parent=[hierarchy_frozen["parent"]] * 5)),
        scaling_siblings=method("scaling_siblings", "global_scaled", "hierarchy", "global",
                                **hierarchy_settings(parent=[scaling_frozen["selected_parent"]] * 5,
                                                     gamma=[scaling_frozen["gamma"]] * 5, scaled=True)),
        decay=method("decay", "global_combined_decay", "hierarchy", "global",
                     beta=combined["beta"], parent=combined["parent"], gamma=combined["gamma"],
                     adaptive=combined["adaptive"], table=combined["table"]),
        nucleus_pairwise=method("nucleus_pairwise", "global_" + instances_frozen["name"], "nucleus_pairwise",
                                "global", unary=instances_frozen["unary"], table=instances_frozen["table"],
                                beta=instances_frozen["beta"], area_weight=instances_frozen["area_weight"]),
        type_counts=method("type_counts", "global_" + presence_frozen["count"]["name"], "presence", "global",
                           **without_name(presence_frozen["count"])),
        type_set=method("type_set", "global_" + presence_frozen["presence"]["name"], "presence", "global",
                        **without_name(presence_frozen["presence"])),
        raw_component=method("raw_component", "global_component", "component", "global", **component_frozen),
    )


HIERARCHY_GRAPH = hierarchy_core.graph()


# ---------------------------------------------------------------- scoring

def score_methods(methods, patches, tables_by_source, label):
    """Return {method name: per-patch 6x6 confusion matrices}."""
    matrices = {}
    started = time.perf_counter()
    for number, item in enumerate(methods, start=1):
        per_patch = []
        for patch in patches:
            prediction = predict(item, patch, tables_by_source)
            per_patch.append(confusion(patch["gt"], prediction))
        matrices[item["name"]] = np.stack(per_patch)
        if number % 10 == 0 or number == len(methods):
            elapsed = time.perf_counter() - started
            print(f"{label}: {number}/{len(methods)} methods scored ({elapsed:.0f} s)", flush=True)
    return matrices


def pick_winner(candidates, scores):
    best_score = max(scores[item["name"]] for item in candidates)
    for item in candidates:
        if scores[item["name"]] >= best_score - TIE_THRESHOLD:
            return item
    raise AssertionError("unreachable")


def combined_decay(winners_by_prefix):
    """family_decay_20261117's combined schedule: best parent, sibling and unary schedules together."""
    parent = winners_by_prefix["parent"]["params"]
    sibling = winners_by_prefix["sibling"]["params"]
    unary = winners_by_prefix["unary"]["params"]
    return method("decay", "decay_combined", "hierarchy", beta=sibling["beta"], parent=parent["parent"],
                  gamma=unary["gamma"], adaptive=unary["adaptive"], table="hybrid")


def develop(organ, folds=None):
    out = output_dir(organ)
    assert not (out / "frozen.json").exists(), "frozen.json already exists: do not retune."
    split = read_json(out / "split.json")
    organ_tables = fit_everything(split)
    save_tables(out, organ_tables)
    tables_by_source = dict(organ=organ_tables)

    patches = load_patches(split["dev"], organ_tables)
    grid = development_grid()
    all_candidates = [item for family in FAMILIES for item in grid[family]]
    matrices = score_methods(list(controls().values()) + all_candidates, patches, tables_by_source, "dev")
    scores = {name: biological_miou(per_patch.sum(0)) for name, per_patch in matrices.items()}

    decay_leaders = {}
    for prefix in ("parent", "sibling", "unary"):
        family_members = [item for item in grid["decay"] if item["name"].startswith(prefix + "_")]
        decay_leaders[prefix] = max(family_members, key=lambda item: scores[item["name"]])
    combined = combined_decay(decay_leaders)
    combined_matrices = score_methods([combined], patches, tables_by_source, "dev combined decay")
    matrices.update(combined_matrices)
    scores[combined["name"]] = biological_miou(combined_matrices[combined["name"]].sum(0))
    grid["decay"].append(combined)

    winners = {family: pick_winner(grid[family], scores) for family in FAMILIES}
    rows = []
    for item in list(controls().values()) + [m for family in FAMILIES for m in grid[family]]:
        rows.append(dict(family=item["family"], name=item["name"], dev_bio_miou=scores[item["name"]],
                         params=item["params"]))
    write_json(out / "development.json", dict(n_dev_patches=len(patches), rows=rows))
    write_json(out / "frozen.json", dict(
        winners=winners, controls=controls(), family_control=FAMILY_CONTROL,
        selection=f"Highest dev pooled five-class biological mIoU; within {TIE_THRESHOLD} the earlier "
                  "(simpler or weaker) grid entry wins.",
        dev_scores={family: scores[winners[family]["name"]] for family in FAMILIES},
    ))
    print("\nFrozen winners (dev biological mIoU):", flush=True)
    print(f'  Raw TP {scores["raw_tp"]:.5f}   vote {scores["fixed_mask_vote"]:.5f}', flush=True)
    for family in FAMILIES:
        print(f'  {family:18s} {winners[family]["name"]:45s} {scores[winners[family]["name"]]:.5f}', flush=True)


# ---------------------------------------------------------------- test

def bootstrap_weights(n_patches):
    draws = np.random.default_rng(SEED).integers(n_patches, size=(BOOTSTRAP_REPLICATES, n_patches))
    weights = np.zeros((BOOTSTRAP_REPLICATES, n_patches), dtype=np.int32)
    for replicate in range(BOOTSTRAP_REPLICATES):
        np.add.at(weights[replicate], draws[replicate], 1)
    return weights


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values))
    running_max = 0.0
    for rank, position in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[position])
        running_max = max(running_max, value)
        adjusted[position] = running_max
    return adjusted


def compare(name_a, name_b, matrices, bootstrap_iou):
    point = class_iou(matrices[name_a].sum(0)) - class_iou(matrices[name_b].sum(0))
    samples = bootstrap_iou[name_a] - bootstrap_iou[name_b]
    biological = samples[:, 1:].mean(1)
    no_dead = samples[:, NO_DEAD_CLASSES].mean(1)
    per_patch_a = class_iou(matrices[name_a])[:, 1:].mean(1)
    per_patch_b = class_iou(matrices[name_b])[:, 1:].mean(1)
    patch_delta = per_patch_a - per_patch_b
    fraction_at_or_below_zero = float((biological <= 0).mean())
    fraction_at_or_above_zero = float((biological >= 0).mean())
    p_value = min(1.0, 2 * min(fraction_at_or_below_zero, fraction_at_or_above_zero))
    return dict(
        delta=float(point[1:].mean()), ci95=np.percentile(biological, [2.5, 97.5]).tolist(),
        bootstrap_p=p_value,
        no_dead_delta=float(point[NO_DEAD_CLASSES].mean()),
        no_dead_ci95=np.percentile(no_dead, [2.5, 97.5]).tolist(),
        per_class_delta=dict(zip(CLASS_NAMES[1:], point[1:].tolist())),
        wins_losses_ties=[int((patch_delta > 1e-12).sum()), int((patch_delta < -1e-12).sum()),
                          int((abs(patch_delta) <= 1e-12).sum())],
    )


def test(organ, folds=None):
    out = output_dir(organ)
    assert (out / "frozen.json").exists(), "Run the dev stage first."
    assert not (out / "evaluation.json").exists(), "evaluation.json exists: the test split is scored once."
    split = read_json(out / "split.json")
    frozen = read_json(out / "frozen.json")
    organ_tables = fit_everything(split)
    tables_by_source = dict(organ=organ_tables, **{"global": global_tables()})

    patches = load_patches(split["test"], organ_tables)
    winners = frozen["winners"]
    global_by_family = global_methods()
    methods = list(controls().values()) + [winners[f] for f in FAMILIES] + [global_by_family[f] for f in FAMILIES]
    matrices = score_methods(methods, patches, tables_by_source, "test")

    weights = bootstrap_weights(len(patches))
    bootstrap_iou = {}
    for name, per_patch in matrices.items():
        pooled = (weights @ per_patch.reshape(len(patches), -1)).reshape(-1, 6, 6)
        bootstrap_iou[name] = class_iou(pooled)

    control_names = {key: item["name"] for key, item in controls().items()}
    families = {}
    for family in FAMILIES:
        winner_name = winners[family]["name"]
        global_name = global_by_family[family]["name"]
        control_name = control_names[FAMILY_CONTROL[family]]
        families[family] = dict(
            organ_winner=winners[family], global_config=global_by_family[family], control=control_name,
            organ_bio_miou=biological_miou(matrices[winner_name].sum(0)),
            global_bio_miou=biological_miou(matrices[global_name].sum(0)),
            control_bio_miou=biological_miou(matrices[control_name].sum(0)),
            organ_vs_control=compare(winner_name, control_name, matrices, bootstrap_iou),
            organ_vs_raw=compare(winner_name, "raw_tp", matrices, bootstrap_iou),
            organ_vs_global=compare(winner_name, global_name, matrices, bootstrap_iou),
            global_vs_control=compare(global_name, control_name, matrices, bootstrap_iou),
        )

    adjusted = holm_adjust(np.array([families[f]["organ_vs_control"]["bootstrap_p"] for f in FAMILIES]))
    for family, p_value in zip(FAMILIES, adjusted):
        result = families[family]
        result["organ_vs_control"]["holm_p"] = float(p_value)
        result["works"] = bool(p_value < 0.05 and result["organ_vs_control"]["delta"] > 0)
        result["beats_global"] = bool(result["organ_vs_global"]["ci95"][0] > 0)

    # Per-fold point estimates: HoVer-Net may have trained on some folds, which would
    # inflate raw scores there. Paired WCSP-vs-control gains should be comparable.
    by_fold = {}
    patch_folds = np.array([row.get("fold", 2) for row in split["test"]])
    for fold in sorted(set(patch_folds.tolist())):
        chosen = patch_folds == fold
        fold_result = dict(n_patches=int(chosen.sum()),
                           raw_tp_bio_miou=biological_miou(matrices["raw_tp"][chosen].sum(0)),
                           vote_bio_miou=biological_miou(matrices["fixed_mask_vote"][chosen].sum(0)),
                           organ_minus_control={})
        for family in FAMILIES:
            winner_score = biological_miou(matrices[winners[family]["name"]][chosen].sum(0))
            control_name = control_names[FAMILY_CONTROL[family]]
            control_score = biological_miou(matrices[control_name][chosen].sum(0))
            fold_result["organ_minus_control"][family] = winner_score - control_score
        by_fold[str(fold)] = fold_result

    raw_iou = class_iou(matrices["raw_tp"].sum(0))
    write_json(out / "evaluation.json", dict(
        by_fold=by_fold,
        organ=organ, n_test_patches=len(patches),
        hovernet_raw_tp=dict(bio_miou=float(raw_iou[1:].mean()), per_class_iou=dict(zip(CLASS_NAMES, raw_iou.tolist()))),
        fixed_mask_vote_bio_miou=biological_miou(matrices["fixed_mask_vote"].sum(0)),
        families=families,
        works_definition="Holm-adjusted (8 families) two-sided paired-bootstrap p < 0.05 against the matched control, with a positive gain.",
        bootstrap=dict(replicates=BOOTSTRAP_REPLICATES, seed=SEED, unit="whole patches; matrices pooled before IoU"),
    ))
    np.savez_compressed(out / "confusions.npz", **matrices)

    with (out / "per_patch.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        names = list(matrices)
        writer.writerow(["image_index", "fold", *names])
        for position, patch in enumerate(patches):
            row = [patch["index"], patch_folds[position]]
            for name in names:
                row.append(float(class_iou(matrices[name][position])[1:].mean()))
            writer.writerow(row)

    print(f"\n{organ}: {len(patches)} test patches, Raw TP {raw_iou[1:].mean():.5f}", flush=True)
    for family in FAMILIES:
        result = families[family]
        versus_control = result["organ_vs_control"]
        versus_global = result["organ_vs_global"]
        print(f'  {family:18s} vs {result["control"]:16s} {versus_control["delta"]:+.5f} '
              f'[{versus_control["ci95"][0]:+.5f}, {versus_control["ci95"][1]:+.5f}] holm p={versus_control["holm_p"]:.3f}'
              f'   vs global {versus_global["delta"]:+.5f}   works={result["works"]}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "infer", "dev", "test"])
    parser.add_argument("--organ", required=True, help="PanNuke tissue name, e.g. Adrenal_gland, Breast, Colon")
    parser.add_argument("--folds", default="2", help="Comma-separated PanNuke folds for the split (prepare only), e.g. 1,2,3")
    arguments = parser.parse_args()
    stages = dict(prepare=prepare, infer=infer, dev=develop, test=test)
    folds = [int(fold) for fold in arguments.folds.split(",")]
    stages[arguments.stage](arguments.organ, folds)
