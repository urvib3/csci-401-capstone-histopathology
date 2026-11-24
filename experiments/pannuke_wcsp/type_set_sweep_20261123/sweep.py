"""Wider sweep of the PanNuke type-set WCSP on fresh images from folds 1 and 3.

The type-set cost: for each image, consider every combination of nucleus types
that could be present. Each combination costs -log(how often that combination
appeared in the fit images), times a strength. WCSP picks the labeling whose
HoVer-Net cost plus this combination cost is lowest.

This sweep varies every knob of that cost (strength, how the per-nucleus cost is
read from HoVer-Net, the evidence penalty, nucleus-count groups, smoothing, and
per-organ counting), picks one setting on dev images and scores it once on test
images that no earlier experiment used.

Stages, run from the repository root in order:
    .venv/bin/python experiments/pannuke_wcsp/type_set_sweep_20261123/sweep.py prepare
    .venv/bin/python experiments/pannuke_wcsp/type_set_sweep_20261123/sweep.py infer
    .venv/bin/python experiments/pannuke_wcsp/type_set_sweep_20261123/sweep.py dev
    .venv/bin/python experiments/pannuke_wcsp/type_set_sweep_20261123/sweep.py test
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "organ_wcsp"))
import run_all as organ  # noqa: E402  (loads the original type-set and instance modules)

presence = organ.presence
nucleus = organ.nucleus
source = organ.source

CACHE = HERE / "cache"
nucleus.CACHE = CACHE
presence.CACHE = CACHE
source.CACHE = CACHE
organ.raw_star.CACHE = CACHE
source.HERE = HERE

SEED = 20261123
FOLDS = [1, 3]
TEST_SIZE = 600
DEV_SIZE = 400
# Every Adrenal_gland image in folds 1 and 3 was used by the Adrenal per-organ run.
EXCLUDED_ORGANS = ["Adrenal_gland"]
BOOTSTRAP_REPLICATES = 5000
TIE_THRESHOLD = 0.0001
SET_COUNT = 1 << presence.N

STRENGTHS = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
UNARIES = ["vote", "mean", "log_mean"]
MODES = ["presence", "presence_evidence"]
BAND_OPTIONS = {"none": [], "original": [8, 20, 40], "finer": [5, 10, 15, 20, 30, 40, 60]}
SMOOTHING = [1, 4, 16, 64]
ORGAN_SPECIFIC = [False, True]

# The frozen setting from presence_20261114, which used 200 fold 2 fit images.
ORIGINAL = dict(mode="presence_evidence", unary="vote", strength=1.0, bands="original",
                smoothing=16, organ_specific=False)
REPORTED_GROUPS = ["Breast", "Colon"]


# ---------------------------------------------------------------- split and data

def make_split():
    pool = []
    for fold in FOLDS:
        tissues = organ.fold_tissues(fold)
        for index, tissue in enumerate(tissues):
            if str(tissue) in EXCLUDED_ORGANS:
                continue
            pool.append(dict(image_index=organ.patch_id(fold, index), fold=fold, index_in_fold=index,
                             tissue=str(tissue)))

    order = np.random.default_rng(SEED).permutation(len(pool))
    test = [pool[i] for i in order[:TEST_SIZE]]
    dev = [pool[i] for i in order[TEST_SIZE:TEST_SIZE + DEV_SIZE]]
    fit = [pool[i] for i in order[TEST_SIZE + DEV_SIZE:]]
    for group in (fit, dev, test):
        group.sort(key=lambda row: row["image_index"])

    split = dict(
        folds=FOLDS, seed=SEED, excluded_organs=EXCLUDED_ORGANS,
        sizes=dict(fit=len(fit), dev=len(dev), test=len(test)),
        rule="Shuffle every fold 1 and 3 image (minus excluded organs) once; first 600 test, next 400 dev, rest fit.",
        patch_id_rule="image_index = fold offset + index within fold; offsets " + str(organ.FOLD_OFFSETS),
        independence="Patch-disjoint only. PanNuke has no patient or slide IDs. "
                     "The folds HoVer-Net was trained on are unknown.",
        fit=fit, dev=dev, test=test, scored=dev + test,
    )
    organ.write_json(HERE / "split.json", split)
    return split


def extract_ground_truth(rows):
    by_fold = {}
    for row in rows:
        if not (CACHE / f'gt_{row["image_index"]:04d}.npz').exists():
            by_fold.setdefault(row["fold"], {})[row["index_in_fold"]] = row["image_index"]
    for fold, id_by_index in by_fold.items():
        print(f"Fold {fold}: extracting ground truth for {len(id_by_index)} images", flush=True)
        for index, masks in organ.stream_fold_arrays(fold, "masks.npy", id_by_index):
            instances, types = source.ground_truth(masks)
            np.savez_compressed(CACHE / f"gt_{id_by_index[index]:04d}.npz", instances=instances, types=types)


def extract_images(rows):
    by_fold = {}
    for row in rows:
        if not (CACHE / f'rgb_{row["image_index"]:04d}.npy').exists():
            by_fold.setdefault(row["fold"], {})[row["index_in_fold"]] = row["image_index"]
    for fold, id_by_index in by_fold.items():
        print(f"Fold {fold}: extracting RGB for {len(id_by_index)} images", flush=True)
        for index, image in organ.stream_fold_arrays(fold, "images.npy", id_by_index):
            converted = image.astype(np.uint8)
            assert np.array_equal(image, converted)
            np.save(CACHE / f'rgb_{id_by_index[index]:04d}.npy', converted)


def development_grid():
    """Every setting, each with a simplicity key used only to break near-ties."""
    grid = []
    for organ_specific in ORGAN_SPECIFIC:
        for band_name, bands in BAND_OPTIONS.items():
            for smoothing in SMOOTHING:
                for mode in MODES:
                    for unary in UNARIES:
                        for strength in STRENGTHS:
                            config = dict(mode=mode, unary=unary, strength=strength, bands=band_name,
                                          smoothing=smoothing, organ_specific=organ_specific)
                            config["name"] = config_name(config)
                            grid.append(config)
    return grid


def config_name(config):
    organ_part = "organ" if config["organ_specific"] else "all"
    return (f'{config["unary"]}_{config["mode"]}_{config["strength"]:g}'
            f'_bands-{config["bands"]}_smooth{config["smoothing"]}_{organ_part}')


def simplicity(config):
    """Smaller is simpler: closer to the original setting, then weaker."""
    return (
        config["organ_specific"],
        len(BAND_OPTIONS[config["bands"]]) != len(BAND_OPTIONS["original"]),
        config["strength"],
        config["mode"] != ORIGINAL["mode"],
        UNARIES.index(config["unary"]),
        abs(np.log(config["smoothing"] / ORIGINAL["smoothing"])),
    )


def prepare():
    CACHE.mkdir(exist_ok=True)
    if (HERE / "split.json").exists():
        split = organ.read_json(HERE / "split.json")
        print("Using the existing split.json", flush=True)
    else:
        split = make_split()
    print("Sizes:", split["sizes"], flush=True)

    if not (HERE / "protocol.json").exists():
        grid = development_grid()
        organ.write_json(HERE / "protocol.json", dict(
            question="Does a wider sweep of the type-set WCSP beat the original frozen setting on fresh images?",
            data="PanNuke folds 1 and 3, all organs except Adrenal_gland. Fit: ground truth only. "
                 "Dev and test: frozen HoVer-Net, full 256x256 patches, same inference as instances_20261114.",
            knobs=dict(strength=STRENGTHS, unary=UNARIES, mode=MODES, bands=BAND_OPTIONS, smoothing=SMOOTHING,
                       organ_specific=ORGAN_SPECIFIC),
            grid_size=len(grid),
            original_setting=ORIGINAL,
            organ_specific_rule="Counts per organ, smoothed toward the all-organ count for the same nucleus-count "
                                "group with the same smoothing weight. Organs missing from fit use all-organ counts.",
            selection=f"Highest pooled five-class biological mIoU on dev. Within {TIE_THRESHOLD}, the simplest "
                      "setting wins: all-organ before per-organ, original groups before others, weaker strength, "
                      "original mode, then vote/mean/log_mean, then smoothing closest to 16.",
            primary=["sweep winner vs. fixed-mask vote", "sweep winner vs. original setting with original counts"],
            primary_test="Two-sided paired bootstrap over whole images, 5000 replicates, seed 20261123, "
                         "confusion matrices pooled before IoU; Holm across the two primary comparisons; "
                         "'works' means Holm p < 0.05 with a positive gain.",
            exploratory=["original setting refit on the new fit images vs. original counts (more data)",
                         "sweep winner vs. original setting refit (better knobs)",
                         "Breast-only and Colon-only test images",
                         "test curve over strength with the winner's other knobs fixed"],
            written="Before any dev or test scoring.",
        ))

    # Ground truth for the original 200 fit images, to check this file's counting reproduces priors.json.
    original_fit = []
    for row in organ.read_json(organ.PRESENCE_DIR / "split.json")["train"]:
        original_fit.append(dict(image_index=row["image_index"], fold=2, index_in_fold=row["image_index"]))
    extract_ground_truth(original_fit + split["fit"] + split["scored"])
    extract_images(split["scored"])
    check_original_counts()
    print("Prepared", flush=True)


def infer():
    """Frozen checkpoint, same full-patch tiling and watershed as instances_20261114/prepare.py."""
    source.infer("scored")


# ---------------------------------------------------------------- counting the type-set frequencies

def nucleus_labels(records):
    label_lists = []
    for row in records:
        _, types = nucleus.load_gt(row["image_index"])
        label_lists.append(types[1:].astype(int) - 1)
    return label_lists


def type_mask(labels):
    mask = 0
    for label in labels:
        mask = mask | (1 << int(label))
    return mask


def fit_prior(label_lists, tissues, bands, smoothing, organ_specific):
    """Probability of each type combination, per nucleus-count group (and per organ if asked)."""
    group_count = len(bands) + 1
    counts = np.zeros((group_count, SET_COUNT))
    organ_counts = {}
    for labels, tissue in zip(label_lists, tissues):
        group = np.searchsorted(bands, len(labels), side="right")
        mask = type_mask(labels)
        counts[group, mask] += 1
        if tissue not in organ_counts:
            organ_counts[tissue] = np.zeros((group_count, SET_COUNT))
        organ_counts[tissue][group, mask] += 1

    overall = (counts.sum(0) + 1) / (counts.sum() + SET_COUNT)
    by_group = (counts + smoothing * overall) / (counts.sum(1, keepdims=True) + smoothing)
    prior = {"all": by_group}
    if organ_specific:
        for tissue, tissue_counts in organ_counts.items():
            prior[tissue] = (tissue_counts + smoothing * by_group) / (tissue_counts.sum(1, keepdims=True) + smoothing)
    return prior


def prior_key(config):
    return (config["bands"], config["smoothing"], config["organ_specific"])


def fit_all_priors(fit_records):
    label_lists = nucleus_labels(fit_records)
    tissues = [row["tissue"] for row in fit_records]
    priors = {}
    for organ_specific in ORGAN_SPECIFIC:
        for band_name, bands in BAND_OPTIONS.items():
            for smoothing in SMOOTHING:
                priors[(band_name, smoothing, organ_specific)] = fit_prior(
                    label_lists, tissues, bands, smoothing, organ_specific)
    return priors


def original_priors():
    """The counts frozen by presence_20261114 (200 fold 2 images), keyed like a refit prior."""
    band_probability = np.array(organ.read_json(organ.PRESENCE_DIR / "priors.json")["band_probability"])
    return {"all": band_probability}


def check_original_counts():
    """fit_prior on the original 200 fit images must reproduce the original priors.json exactly."""
    original_fit = organ.read_json(organ.PRESENCE_DIR / "split.json")["train"]
    label_lists = nucleus_labels(original_fit)
    refit = fit_prior(label_lists, ["unused"] * len(label_lists), BAND_OPTIONS["original"], 16, False)
    np.testing.assert_allclose(refit["all"], original_priors()["all"], rtol=0, atol=1e-12)
    print("Counting reproduces presence_20261114/priors.json", flush=True)


# ---------------------------------------------------------------- scoring

def load_patches(records):
    placeholder_band = np.full((len(presence.BANDS) + 1, SET_COUNT), 1 / SET_COUNT)
    placeholder_frequency = np.full(presence.N, 1 / presence.N)
    patches = list(presence.records(records, placeholder_band, placeholder_frequency))
    for patch, row in zip(patches, records):
        patch["tissue"] = row["tissue"]
        patch["confusion_cache"] = {}
    return patches


def set_probability(patch, prior, bands):
    table = prior.get(patch["tissue"], prior["all"])
    group = np.searchsorted(bands, len(patch["baseline"]), side="right")
    return table[group, presence.MASKS]


def chosen_labels(patch, config, prior):
    """Same objective as presence_20261114/run.py predict(), with this prior swapped in."""
    if len(patch["baseline"]) == 0:
        return None, patch["baseline"]
    labels, minima = patch["options"][config["unary"]]
    probability = set_probability(patch, prior, BAND_OPTIONS[config["bands"]])
    cost = minima - config["strength"] * np.log(probability)
    if config["mode"] == "presence_evidence":
        cost = cost + config["strength"] * patch["evidence_penalty"]
    chosen = int(cost.argmin())
    return chosen, labels[chosen]


def config_confusions(patches, config, prior):
    per_patch = []
    for patch in patches:
        chosen, labels = chosen_labels(patch, config, prior)
        key = (config["unary"], chosen)
        if key not in patch["confusion_cache"]:
            image = organ.nucleus_labels_to_image(patch, labels)
            patch["confusion_cache"][key] = organ.confusion(patch["gt"], image)
        per_patch.append(patch["confusion_cache"][key])
    return np.stack(per_patch)


def control_confusions(patches):
    raw, vote = [], []
    for patch in patches:
        raw.append(organ.confusion(patch["gt"], patch["raw"]))
        vote.append(organ.confusion(patch["gt"], organ.nucleus_labels_to_image(patch, patch["baseline"])))
    return dict(raw_tp=np.stack(raw), fixed_mask_vote=np.stack(vote))


def check_against_original_solver(patches, prior):
    """chosen_labels must match presence.predict for the original setting on every patch."""
    config = dict(ORIGINAL)
    for patch in patches:
        adjusted = dict(patch, set_probability=set_probability(patch, prior, BAND_OPTIONS["original"]))
        expected = presence.predict(config, adjusted)
        _, labels = chosen_labels(patch, config, prior)
        np.testing.assert_array_equal(labels, expected)
    print(f"Matches presence_20261114 predict() on {len(patches)} images", flush=True)


def develop():
    assert not (HERE / "frozen.json").exists(), "frozen.json exists: do not retune."
    split = organ.read_json(HERE / "split.json")
    started = time.perf_counter()
    priors = fit_all_priors(split["fit"])
    patches = load_patches(split["dev"])
    print(f"Loaded {len(patches)} dev images ({time.perf_counter() - started:.0f} s)", flush=True)
    check_against_original_solver(patches, original_priors())

    controls = control_confusions(patches)
    scores = {name: organ.biological_miou(matrices.sum(0)) for name, matrices in controls.items()}
    original_old = config_confusions(patches, ORIGINAL, original_priors())
    scores["original_setting_original_counts"] = organ.biological_miou(original_old.sum(0))

    grid = development_grid()
    rows = []
    for number, config in enumerate(grid, start=1):
        matrices = config_confusions(patches, config, priors[prior_key(config)])
        score = organ.biological_miou(matrices.sum(0))
        scores[config["name"]] = score
        rows.append(dict(config=config, dev_bio_miou=score))
        if number % 200 == 0 or number == len(grid):
            print(f"dev: {number}/{len(grid)} settings ({time.perf_counter() - started:.0f} s)", flush=True)

    best = max(row["dev_bio_miou"] for row in rows)
    near_best = [row["config"] for row in rows if row["dev_bio_miou"] >= best - TIE_THRESHOLD]
    winner = min(near_best, key=simplicity)
    original_refit_name = config_name(ORIGINAL)

    best_by_strength = {}
    for strength in STRENGTHS:
        candidates = [row for row in rows if row["config"]["strength"] == strength]
        top = max(candidates, key=lambda row: row["dev_bio_miou"])
        best_by_strength[str(strength)] = dict(dev_bio_miou=top["dev_bio_miou"], config=top["config"])

    organ.write_json(HERE / "development.json", dict(
        n_dev_images=len(patches), controls={name: scores[name] for name in
                                             ("raw_tp", "fixed_mask_vote", "original_setting_original_counts")},
        best_by_strength=best_by_strength, rows=rows))
    organ.write_json(HERE / "frozen.json", dict(
        winner=winner, winner_dev_bio_miou=scores[winner["name"]], settings_within_tie=len(near_best),
        original_setting=ORIGINAL, original_refit_dev_bio_miou=scores[original_refit_name],
        dev_controls={name: scores[name] for name in
                      ("raw_tp", "fixed_mask_vote", "original_setting_original_counts")},
        written_before_test=True,
    ))
    print(f'\nDev: Raw TP {scores["raw_tp"]:.5f}  vote {scores["fixed_mask_vote"]:.5f}  '
          f'original {scores["original_setting_original_counts"]:.5f}  '
          f'original refit {scores[original_refit_name]:.5f}', flush=True)
    print(f'Frozen winner {winner["name"]}: {scores[winner["name"]]:.5f} '
          f'({len(near_best)} settings within {TIE_THRESHOLD})', flush=True)
    for strength, entry in best_by_strength.items():
        print(f'  strength {strength:>5}: best dev {entry["dev_bio_miou"]:.5f}  {entry["config"]["name"]}', flush=True)


def bootstrap_weights(n_images, seed):
    draws = np.random.default_rng(seed).integers(n_images, size=(BOOTSTRAP_REPLICATES, n_images))
    weights = np.zeros((BOOTSTRAP_REPLICATES, n_images), dtype=np.int32)
    for replicate in range(BOOTSTRAP_REPLICATES):
        np.add.at(weights[replicate], draws[replicate], 1)
    return weights


def bootstrap_class_iou(matrices, weights):
    result = {}
    for name, per_patch in matrices.items():
        pooled = (weights @ per_patch.reshape(len(per_patch), -1)).reshape(-1, 6, 6)
        result[name] = organ.class_iou(pooled)
    return result


def test():
    assert (HERE / "frozen.json").exists(), "Run the dev stage first."
    assert not (HERE / "evaluation.json").exists(), "evaluation.json exists: the test split is scored once."
    split = organ.read_json(HERE / "split.json")
    frozen = organ.read_json(HERE / "frozen.json")
    winner = frozen["winner"]
    priors = fit_all_priors(split["fit"])
    patches = load_patches(split["test"])

    matrices = control_confusions(patches)
    matrices["original_setting_original_counts"] = config_confusions(patches, ORIGINAL, original_priors())
    matrices["original_setting_refit"] = config_confusions(patches, ORIGINAL, priors[prior_key(ORIGINAL)])
    matrices["sweep_winner"] = config_confusions(patches, winner, priors[prior_key(winner)])

    weights = bootstrap_weights(len(patches), SEED)
    bootstrap_iou = bootstrap_class_iou(matrices, weights)

    def compare(name_a, name_b):
        return organ.compare(name_a, name_b, matrices, bootstrap_iou)

    primary = {
        "sweep_winner vs fixed_mask_vote": compare("sweep_winner", "fixed_mask_vote"),
        "sweep_winner vs original_setting_original_counts": compare("sweep_winner", "original_setting_original_counts"),
    }
    adjusted = organ.holm_adjust(np.array([row["bootstrap_p"] for row in primary.values()]))
    for row, p_value in zip(primary.values(), adjusted):
        row["holm_p"] = float(p_value)
        row["works"] = bool(p_value < 0.05 and row["delta"] > 0)

    exploratory = {}
    for name_a, name_b in [("sweep_winner", "raw_tp"),
                           ("original_setting_original_counts", "fixed_mask_vote"),
                           ("original_setting_original_counts", "raw_tp"),
                           ("original_setting_refit", "original_setting_original_counts"),
                           ("sweep_winner", "original_setting_refit"),
                           ("fixed_mask_vote", "raw_tp")]:
        exploratory[f"{name_a} vs {name_b}"] = compare(name_a, name_b)

    by_group = {}
    for group in REPORTED_GROUPS + ["Other organs"]:
        positions = []
        for position, patch in enumerate(patches):
            in_group = patch["tissue"] == group if group != "Other organs" else patch["tissue"] not in REPORTED_GROUPS
            if in_group:
                positions.append(position)
        subset = {name: per_patch[positions] for name, per_patch in matrices.items()}
        subset_iou = bootstrap_class_iou(subset, bootstrap_weights(len(positions), SEED))
        by_group[group] = dict(
            n_images=len(positions),
            bio_miou={name: organ.biological_miou(per_patch.sum(0)) for name, per_patch in subset.items()},
            sweep_winner_vs_vote=organ.compare("sweep_winner", "fixed_mask_vote", subset, subset_iou),
            original_vs_vote=organ.compare("original_setting_original_counts", "fixed_mask_vote", subset, subset_iou),
        )

    # Exploratory, after freezing: the winner's other knobs, every strength, scored on test.
    vote_score = organ.biological_miou(matrices["fixed_mask_vote"].sum(0))
    strength_curve = {}
    for strength in STRENGTHS:
        config = dict(winner, strength=strength)
        score = organ.biological_miou(config_confusions(patches, config, priors[prior_key(config)]).sum(0))
        strength_curve[str(strength)] = dict(test_bio_miou=score, gain_over_vote=score - vote_score)

    results = {}
    for name, per_patch in matrices.items():
        pooled = per_patch.sum(0)
        results[name] = dict(bio_miou=organ.biological_miou(pooled),
                             no_dead_miou=float(organ.class_iou(pooled)[organ.NO_DEAD_CLASSES].mean()),
                             per_class_iou=dict(zip(organ.CLASS_NAMES[1:], organ.class_iou(pooled)[1:].tolist())))

    organ.write_json(HERE / "evaluation.json", dict(
        n_test_images=len(patches), winner=winner, results=results, primary=primary,
        exploratory=exploratory, by_group=by_group, test_strength_curve_exploratory=strength_curve,
        bootstrap=dict(replicates=BOOTSTRAP_REPLICATES, seed=SEED, unit="whole image"),
    ))
    np.savez_compressed(HERE / "confusions.npz", **matrices)
    with (HERE / "per_patch.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_index", "fold", "tissue", *matrices])
        for position, row in enumerate(split["test"]):
            scores = []
            for per_patch in matrices.values():
                scores.append(float(organ.class_iou(per_patch[position])[1:].mean()))
            writer.writerow([row["image_index"], row["fold"], row["tissue"], *scores])

    for name, result in results.items():
        print(f'{name:36s} {result["bio_miou"]:.5f}', flush=True)
    for label, row in {**primary, **exploratory}.items():
        low, high = row["ci95"]
        holm = f'  Holm p {row["holm_p"]:.4f}' if "holm_p" in row else ""
        print(f'{label:62s} {row["delta"]:+.5f} [{low:+.5f}, {high:+.5f}]{holm}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "infer", "dev", "test"])
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare()
    elif stage == "infer":
        infer()
    elif stage == "dev":
        develop()
    else:
        test()
