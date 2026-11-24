"""PanNuke preliminary tests for Deliverable 7: where HoVer-Net goes wrong and what WCSP could fix.

Mirrors the BCSS preliminary tests (deliverables/deliverable_7.md, section 3):
  3a. which nucleus kinds get confused,
  3b. outline errors (WCSP cannot fix) vs. kind errors (WCSP can),
  3c. best-case ceilings for relabeling and for a perfect "which kinds are here" rule,
  rule check: how often the truth vs. HoVer-Net breaks candidate rules,
  pictures of Breast images before and after type-set WCSP.

Uses the 400 knob-picking (dev) images and the 4,089 counting images of
type_set_sweep_20261123. The 600 final-score images are not touched.

    .venv/bin/python experiments/pannuke_wcsp/prelim_20261123/analyze.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE.parent / "type_set_sweep_20261123"
sys.path.insert(0, str(SWEEP_DIR))
import sweep  # noqa: E402  (also loads organ_wcsp/run_all.py and the original modules)

organ = sweep.organ
KINDS = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Normal epithelial"]
SHORT = ["Neo", "Inf", "Con", "Dead", "Epi"]

# Nucleus kind colors: validated for adjacent pairs; Dead is an intentional neutral gray.
KIND_COLORS = ["#e34948", "#2a78d6", "#1baf7a", "#55554f", "#eda100"]
BACKGROUND = "#f4f3ef"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#55554f"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
ACCENT_LIGHT = "#86b6ef"
SECOND = "#eb6834"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RARE_SET = 0.01


def style_axis(axis):
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID_COLOR)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def new_figure(width, height):
    figure = plt.figure(figsize=(width, height), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    return figure


# ---------------------------------------------------------------- per-image labelings

def oracle_kind_labels(patch):
    """Each predicted nucleus gets the kind most of its pixels have in the truth (outlines unchanged)."""
    instances = patch["instances"]
    nuclei = len(patch["baseline"])
    cross = np.bincount((instances.astype(np.int64) * 6 + patch["gt"]).ravel(),
                        minlength=(nuclei + 1) * 6).reshape(nuclei + 1, 6)
    labels = patch["baseline"].copy()
    for nucleus_id in range(1, nuclei + 1):
        kind_pixels = cross[nucleus_id, 1:]
        if kind_pixels.sum() > 0:
            labels[nucleus_id - 1] = int(kind_pixels.argmax())
    return labels


def oracle_set_labels(patch, unary):
    """Best labeling that uses exactly the kinds present in the truth: a perfect type-set rule."""
    true_mask = sweep.type_mask(patch["truth"])
    labels, minima = patch["options"][unary]
    if true_mask == 0 or not np.isfinite(minima[true_mask - 1]):
        return patch["baseline"]
    return labels[true_mask - 1]


def kinds_in(labels):
    present = np.zeros(5, bool)
    for label in labels:
        present[int(label)] = True
    return present


# ---------------------------------------------------------------- analysis

def main():
    split = organ.read_json(SWEEP_DIR / "split.json")
    frozen = organ.read_json(SWEEP_DIR / "frozen.json")
    winner = frozen["winner"]
    print("Counting priors on", len(split["fit"]), "images", flush=True)
    priors = sweep.fit_all_priors(split["fit"])
    winner_prior = priors[sweep.prior_key(winner)]
    original_prior = sweep.original_priors()
    patches = sweep.load_patches(split["dev"])
    print("Loaded", len(patches), "dev images", flush=True)

    labelings = {"vote": [], "original": [], "winner": [], "oracle_kind": [], "oracle_set": []}
    for patch in patches:
        labelings["vote"].append(patch["baseline"])
        labelings["original"].append(sweep.chosen_labels(patch, sweep.ORIGINAL, original_prior)[1])
        labelings["winner"].append(sweep.chosen_labels(patch, winner, winner_prior)[1])
        labelings["oracle_kind"].append(oracle_kind_labels(patch))
        labelings["oracle_set"].append(oracle_set_labels(patch, winner["unary"]))

    pixel_confusion = {"raw": np.zeros((6, 6), np.int64)}
    per_image_miou = {}
    for name in labelings:
        pixel_confusion[name] = np.zeros((6, 6), np.int64)
        per_image_miou[name] = []
    for position, patch in enumerate(patches):
        pixel_confusion["raw"] += organ.confusion(patch["gt"], patch["raw"])
        for name, all_labels in labelings.items():
            image = organ.nucleus_labels_to_image(patch, all_labels[position])
            matrix = organ.confusion(patch["gt"], image)
            pixel_confusion[name] += matrix
            per_image_miou[name].append(organ.biological_miou(matrix))

    scores = {name: organ.biological_miou(matrix) for name, matrix in pixel_confusion.items()}
    no_dead = {name: float(organ.class_iou(matrix)[organ.NO_DEAD_CLASSES].mean())
               for name, matrix in pixel_confusion.items()}

    # 3a. Nucleus-level kind confusion among nuclei HoVer-Net found correctly (overlap > 0.5).
    kind_confusion = {"vote": np.zeros((5, 5), np.int64), "winner": np.zeros((5, 5), np.int64)}
    true_nuclei = matched_nuclei = predicted_nuclei = 0
    for position, patch in enumerate(patches):
        true_nuclei += len(patch["truth"])
        predicted_nuclei += len(patch["baseline"])
        matched_nuclei += len(patch["ts"])
        for name in kind_confusion:
            predicted = labelings[name][position][patch["ps"]]
            for truth_kind, predicted_kind in zip(patch["ts"], predicted):
                kind_confusion[name][int(truth_kind), int(predicted_kind)] += 1

    # 3b. Where the wrong pixels come from (voting output).
    vote = pixel_confusion["vote"]
    wrong = vote.sum() - np.trace(vote)
    missed = vote[1:, 0].sum()
    extra = vote[0, 1:].sum()
    wrong_kind = wrong - missed - extra
    oracle = pixel_confusion["oracle_kind"]
    kind_left_after_oracle = (oracle.sum() - np.trace(oracle)) - oracle[1:, 0].sum() - oracle[0, 1:].sum()
    error_split = dict(
        wrong_pixels=int(wrong), pixel_error_rate=float(wrong / vote.sum()),
        missed_nucleus_pixels=float(missed / wrong), extra_nucleus_pixels=float(extra / wrong),
        wrong_kind_pixels=float(wrong_kind / wrong),
        wrong_kind_fixable_by_relabeling=float((wrong_kind - kind_left_after_oracle) / wrong),
        wrong_kind_inside_merged_or_split_nuclei=float(kind_left_after_oracle / wrong),
        true_nuclei=true_nuclei, predicted_nuclei=predicted_nuclei, matched_nuclei=matched_nuclei,
        true_nuclei_found=float(matched_nuclei / true_nuclei),
        matched_kind_accuracy_vote=float(np.trace(kind_confusion["vote"]) / kind_confusion["vote"].sum()),
        matched_kind_accuracy_winner=float(np.trace(kind_confusion["winner"]) / kind_confusion["winner"].sum()),
    )

    # Rule check: share of images breaking each rule, in the truth vs. in predictions.
    fit_labels = sweep.nucleus_labels(split["fit"])
    set_counts = np.zeros(sweep.SET_COUNT)
    for labels in fit_labels:
        set_counts[sweep.type_mask(labels)] += 1
    set_probability = set_counts / set_counts.sum()
    sources = {
        "Fit truth (4,089)": [(labels, row["tissue"]) for labels, row in zip(fit_labels, split["fit"])],
        "Dev truth (400)": [(patch["truth"], patch["tissue"]) for patch in patches],
        "HoVer-Net vote": [(labelings["vote"][i], patches[i]["tissue"]) for i in range(len(patches))],
        "Type-set WCSP": [(labelings["winner"][i], patches[i]["tissue"]) for i in range(len(patches))],
    }
    rule_rows = {}
    for source_name, items in sources.items():
        with_nuclei = [(labels, tissue) for labels, tissue in items if len(labels) > 0]
        n_images = len(with_nuclei)
        rare = lonely = cancer_and_epithelial = 0
        kinds_per_image = []
        present_share = np.zeros(5)
        lonely_by_kind = np.zeros(5)
        breast_cancer_and_epithelial = breast_images = 0
        for labels, tissue in with_nuclei:
            mask = sweep.type_mask(labels)
            present = kinds_in(labels)
            counts = np.bincount(np.asarray(labels, int), minlength=5)
            present_share += present
            kinds_per_image.append(present.sum())
            if set_probability[mask] < RARE_SET:
                rare += 1
            if ((counts == 1) & present).any():
                lonely += 1
            lonely_by_kind += (counts == 1)
            if present[0] and present[4]:
                cancer_and_epithelial += 1
            if tissue == "Breast":
                breast_images += 1
                if present[0] and present[4]:
                    breast_cancer_and_epithelial += 1
        rule_rows[source_name] = dict(
            images=n_images,
            rare_kind_combination=rare / n_images,
            some_kind_has_a_single_nucleus=lonely / n_images,
            single_nucleus_by_kind=dict(zip(KINDS, (lonely_by_kind / n_images).tolist())),
            kinds_per_image=float(np.mean(kinds_per_image)),
            images_containing=dict(zip(KINDS, (present_share / n_images).tolist())),
            cancer_and_epithelial_together=cancer_and_epithelial / n_images,
            breast_cancer_and_epithelial_together=breast_cancer_and_epithelial / max(breast_images, 1),
        )

    common_sets = []
    for mask in np.argsort(-set_counts)[:12]:
        names = [KINDS[k] for k in range(5) if mask >> k & 1] or ["(no nuclei)"]
        common_sets.append(dict(kinds=" + ".join(names), share=float(set_probability[mask])))

    results = dict(
        n_dev_images=len(patches), winner=winner,
        mIoU={name: scores[name] for name in ("raw", "vote", "original", "winner", "oracle_set", "oracle_kind")},
        mIoU_without_dead={name: no_dead[name] for name in ("raw", "vote", "original", "winner", "oracle_set",
                                                            "oracle_kind")},
        per_class_iou={name: dict(zip(KINDS, organ.class_iou(matrix)[1:].tolist()))
                       for name, matrix in pixel_confusion.items()},
        error_split=error_split,
        nucleus_kind_confusion=dict(rows_true_columns_predicted=KINDS,
                                    vote=kind_confusion["vote"].tolist(), winner=kind_confusion["winner"].tolist()),
        pixel_confusion_vote=dict(rows_true_columns_predicted=["Background"] + KINDS, counts=vote.tolist()),
        rule_check=rule_rows, most_common_kind_sets_fit=common_sets,
        notes=dict(
            oracle_kind="Every HoVer-Net nucleus relabeled with the kind most of its pixels have in the truth. "
                        "Outlines unchanged: the best any relabeling method could do.",
            oracle_set=f'Exact type-set solver restricted to the true set of kinds, unary {winner["unary"]}, '
                       "no prior: a perfect 'which kinds are here' rule.",
            matched="A true and a predicted nucleus match when their overlap (IoU) exceeds 0.5.",
            rare=f"A kind combination is rare when under {RARE_SET:.0%} of counting images have exactly it.",
        ),
    )
    organ.write_json(HERE / "results.json", results)

    plot_confusion(kind_confusion["vote"])
    plot_error_split(error_split)
    plot_ladder(scores)
    plot_rule_check(rule_rows)
    plot_breast_examples(patches, labelings, per_image_miou)
    print_summary(results)


# ---------------------------------------------------------------- figures

def plot_confusion(matrix):
    shares = matrix / matrix.sum(1, keepdims=True)
    figure = new_figure(5.6, 4.6)
    axis = figure.add_subplot(1, 1, 1)
    axis.imshow(shares, cmap=ListedColormap(BLUE_RAMP), vmin=0, vmax=1)
    for row in range(5):
        for column in range(5):
            value = shares[row, column]
            color = SURFACE if value > 0.55 else TEXT_PRIMARY
            axis.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=8.5, color=color)
    axis.set_xticks(range(5))
    axis.set_xticklabels(KINDS, rotation=30, ha="right", fontsize=8, color=TEXT_PRIMARY)
    axis.set_yticks(range(5))
    labels = []
    for kind, total in zip(KINDS, matrix.sum(1)):
        labels.append(f"{kind} ({total:,})")
    axis.set_yticklabels(labels, fontsize=8, color=TEXT_PRIMARY)
    axis.set_xlabel("HoVer-Net says", fontsize=9, color=TEXT_SECONDARY)
    axis.set_ylabel("Truth (number of nuclei)", fontsize=9, color=TEXT_SECONDARY)
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.suptitle("Which nucleus kinds HoVer-Net confuses", x=0.02, ha="left", fontsize=11,
                    color=TEXT_PRIMARY, fontweight="bold")
    axis.set_title("Nuclei HoVer-Net found; each row sums to 100%", loc="left", fontsize=8.5, color=TEXT_SECONDARY)
    figure.tight_layout()
    figure.savefig(HERE / "prelim_confusion.png", facecolor=SURFACE)
    plt.close(figure)


def plot_error_split(split):
    parts = [
        ("Nucleus pixels HoVer-Net missed\n(missed nuclei, outlines too small)", split["missed_nucleus_pixels"], "outline"),
        ("Background HoVer-Net called nucleus\n(extra nuclei, outlines too big)", split["extra_nucleus_pixels"], "outline"),
        ("Wrong kind, fixable by\nrelabeling the nucleus", split["wrong_kind_fixable_by_relabeling"], "kind"),
    ]
    figure = new_figure(7.6, 3.6)
    axis = figure.add_subplot(1, 1, 1)
    style_axis(axis)
    for row, (label, share, group) in enumerate(parts):
        y = len(parts) - 1 - row
        color = SECOND if group == "kind" else ACCENT
        axis.barh(y, share * 100, color=color, height=0.6)
        axis.text(share * 100 + 0.8, y, f"{share:.0%}", va="center", fontsize=9, color=TEXT_PRIMARY)
    axis.set_yticks(range(len(parts)))
    axis.set_yticklabels([part[0] for part in reversed(parts)], fontsize=8, color=TEXT_PRIMARY)
    axis.set_xlabel("Share of all wrong pixels (HoVer-Net with voting)", fontsize=9, color=TEXT_SECONDARY)
    axis.set_xlim(0, 100)
    axis.tick_params(axis="y", length=0)
    axis.legend(handles=[Patch(color=ACCENT, label="Outline errors: WCSP can't fix"),
                         Patch(color=SECOND, label="Kind errors: WCSP can fix")],
                loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)
    figure.suptitle(f'Where the wrong pixels come from ({split["pixel_error_rate"]:.1%} of pixels are wrong)',
                    x=0.02, ha="left", fontsize=11, color=TEXT_PRIMARY, fontweight="bold")
    figure.tight_layout()
    figure.savefig(HERE / "prelim_error_split.png", facecolor=SURFACE)
    plt.close(figure)


def plot_ladder(scores):
    steps = [
        ("HoVer-Net, raw pixels", scores["raw"], "actual"),
        ("HoVer-Net, voting", scores["vote"], "actual"),
        ("Type-set WCSP, original setting", scores["original"], "actual"),
        ("Type-set WCSP, swept setting", scores["winner"], "actual"),
        ("Ceiling: perfect 'which kinds are here'", scores["oracle_set"], "ceiling"),
        ("Ceiling: every nucleus the right kind", scores["oracle_kind"], "ceiling"),
    ]
    figure = new_figure(7.6, 3.8)
    axis = figure.add_subplot(1, 1, 1)
    style_axis(axis)
    base = scores["vote"]
    for row, (label, value, group) in enumerate(steps):
        y = len(steps) - 1 - row
        color = ACCENT if group == "actual" else ACCENT_LIGHT
        axis.barh(y, value, color=color, height=0.6)
        gain = (value - base) * 100
        text = f"{value:.3f}" if label == "HoVer-Net, voting" else f"{value:.3f}  ({gain:+.1f})"
        axis.text(value + 0.004, y, text, va="center", fontsize=8.5, color=TEXT_PRIMARY)
    axis.set_yticks(range(len(steps)))
    axis.set_yticklabels([step[0] for step in reversed(steps)], fontsize=8, color=TEXT_PRIMARY)
    axis.set_xlim(0.40, max(scores["oracle_kind"], 0.5) + 0.09)
    axis.set_xlabel("mIoU on 400 knob-picking images (points vs. voting in brackets)", fontsize=9,
                    color=TEXT_SECONDARY)
    axis.tick_params(axis="y", length=0)
    axis.legend(handles=[Patch(color=ACCENT, label="Actual methods"),
                         Patch(color=ACCENT_LIGHT, label="Best case (uses the truth)")],
                loc="upper right", frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)
    figure.suptitle("How much could fixing nucleus kinds gain?", x=0.02, ha="left", fontsize=11,
                    color=TEXT_PRIMARY, fontweight="bold")
    figure.tight_layout()
    figure.savefig(HERE / "prelim_ladder.png", facecolor=SURFACE)
    plt.close(figure)


def plot_rule_check(rule_rows):
    rules = [
        ("Rare kind combination (<1% of images)", "rare_kind_combination"),
        ("Some kind has just one nucleus", "some_kind_has_a_single_nucleus"),
        ("Cancer and normal lining together", "cancer_and_epithelial_together"),
    ]
    source_names = ["Dev truth (400)", "HoVer-Net vote", "Type-set WCSP"]
    colors = ["#55554f", ACCENT, SECOND]
    figure = new_figure(7.6, 3.6)
    axis = figure.add_subplot(1, 1, 1)
    style_axis(axis)
    height = 0.25
    for rule_number, (label, key) in enumerate(rules):
        y = len(rules) - 1 - rule_number
        for source_number, source_name in enumerate(source_names):
            value = rule_rows[source_name][key] * 100
            position = y + (1 - source_number) * height
            axis.barh(position, value, height=height * 0.9, color=colors[source_number])
            axis.text(value + 0.8, position, f"{value:.0f}%", va="center", fontsize=7.5, color=TEXT_PRIMARY)
    axis.set_yticks(range(len(rules)))
    axis.set_yticklabels([rule[0] for rule in reversed(rules)], fontsize=8, color=TEXT_PRIMARY)
    axis.set_xlabel("Share of images", fontsize=9, color=TEXT_SECONDARY)
    axis.tick_params(axis="y", length=0)
    handles = []
    for source_name, color in zip(source_names, colors):
        handles.append(Patch(color=color, label=source_name))
    axis.legend(handles=handles, loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)
    figure.suptitle("Rule check: how often the truth vs. HoVer-Net breaks each pattern", x=0.02, ha="left",
                    fontsize=11, color=TEXT_PRIMARY, fontweight="bold")
    figure.tight_layout()
    figure.savefig(HERE / "prelim_rule_check.png", facecolor=SURFACE)
    plt.close(figure)


def label_image(patch, labels):
    """RGB picture of a nucleus label map: background light, each nucleus colored by kind, outlines dark."""
    colors = np.array([matplotlib.colors.to_rgb(BACKGROUND)] + [matplotlib.colors.to_rgb(c) for c in KIND_COLORS])
    image = organ.nucleus_labels_to_image(patch, labels) if labels is not None else patch["gt"]
    picture = colors[image]
    return picture


def truth_outlines(patch):
    with np.load(sweep.CACHE / f'gt_{patch["index"]:04d}.npz') as file:
        return file["instances"]


def outline_mask(instances):
    edges = np.zeros(instances.shape, bool)
    edges[:-1, :] |= instances[:-1, :] != instances[1:, :]
    edges[:, :-1] |= instances[:, :-1] != instances[:, 1:]
    return edges & (instances > 0) | np.roll(edges, 1, axis=0) & (instances > 0)


def plot_breast_examples(patches, labelings, per_image_miou):
    breast_positions = []
    for position, patch in enumerate(patches):
        if patch["tissue"] == "Breast":
            breast_positions.append(position)
    gains = []
    for position in breast_positions:
        gains.append(per_image_miou["winner"][position] - per_image_miou["vote"][position])
    order = np.argsort(gains)
    best = [breast_positions[i] for i in order[::-1][:3]]
    worst = [breast_positions[order[0]]]
    chosen = best + worst

    columns = ["Tissue (H&E)", "Truth", "HoVer-Net + voting", "After type-set WCSP"]
    figure = new_figure(9.2, 2.45 * len(chosen) + 0.6)
    for row, position in enumerate(chosen):
        patch = patches[position]
        rgb = np.load(sweep.CACHE / f'rgb_{patch["index"]:04d}.npy')
        truth_picture = label_image(patch, None)
        vote_picture = label_image(patch, labelings["vote"][position])
        winner_picture = label_image(patch, labelings["winner"][position])
        changed = labelings["winner"][position] != labelings["vote"][position]
        changed_ids = np.flatnonzero(changed) + 1
        changed_pixels = np.isin(patch["instances"], changed_ids)
        changed_edges = outline_mask(np.where(changed_pixels, patch["instances"], 0))
        winner_picture[changed_edges] = matplotlib.colors.to_rgb(TEXT_PRIMARY)
        vote_picture[changed_edges] = matplotlib.colors.to_rgb(TEXT_PRIMARY)
        pictures = [rgb, truth_picture, vote_picture, winner_picture]
        found = len(patch["ts"])
        vote_correct = int((labelings["vote"][position][patch["ps"]] == patch["ts"]).sum())
        winner_correct = int((labelings["winner"][position][patch["ps"]] == patch["ts"]).sum())
        changed_count = int(changed.sum())
        nucleus_word = "nucleus" if changed_count == 1 else "nuclei"
        for column, picture in enumerate(pictures):
            axis = figure.add_subplot(len(chosen), 4, row * 4 + column + 1)
            axis.imshow(picture)
            axis.set_xticks([])
            axis.set_yticks([])
            for side in axis.spines.values():
                side.set_visible(False)
            if row == 0:
                axis.set_title(columns[column], fontsize=9, color=TEXT_PRIMARY)
            if column == 2:
                axis.set_xlabel(f"Right kind: {vote_correct} of {found}", fontsize=8, color=TEXT_SECONDARY)
            if column == 3:
                axis.set_xlabel(f"Right kind: {winner_correct} of {found} ({changed_count} {nucleus_word} changed)",
                                fontsize=8, color=TEXT_SECONDARY)
            if column == 0:
                tag = "Biggest loss" if position in worst else "Biggest gain"
                axis.set_ylabel(f'{tag}\nimage {patch["index"]}', fontsize=8, color=TEXT_SECONDARY)
    handles = []
    for kind, color in zip(KINDS, KIND_COLORS):
        handles.append(Patch(color=color, label=kind))
    handles.append(Patch(facecolor=SURFACE, edgecolor=TEXT_PRIMARY, label="Outlined: changed by WCSP"))
    figure.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)
    figure.suptitle("Breast images: what type-set WCSP changes", x=0.02, ha="left", fontsize=11,
                    color=TEXT_PRIMARY, fontweight="bold")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(HERE / "breast_examples.png", facecolor=SURFACE)
    plt.close(figure)
    organ.write_json(HERE / "breast_examples.json", dict(
        chosen_image_ids=[patches[p]["index"] for p in chosen],
        rule="Breast images in the 400 dev images: the 3 with the largest per-image mIoU gain of the swept "
             "type-set WCSP over voting, and the 1 with the largest loss.",
        breast_dev_images=len(breast_positions),
        improved=int(sum(g > 1e-12 for g in gains)), worse=int(sum(g < -1e-12 for g in gains)),
        unchanged=int(sum(abs(g) <= 1e-12 for g in gains)),
    ))


def print_summary(results):
    print("\nmIoU:", {k: round(v, 4) for k, v in results["mIoU"].items()})
    print("Error split:", {k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in results["error_split"].items()})
    for name, row in results["rule_check"].items():
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                     if not isinstance(v, dict)})
    print("Most common kind sets:", results["most_common_kind_sets_fit"][:6])


if __name__ == "__main__":
    main()
