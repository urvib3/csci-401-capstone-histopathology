"""Reproduce Deliverable 6 figures and error audit from cached predictions."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import distance_transform_edt

from evaluate import HERE, CACHE, load_previous, source

ROOT = HERE.parent
COLORS = ["#171717", "#e64b55", "#409fda", "#3eb27d", "#f1ba45", "#bb76d2"]
LABELS = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Non-neoplastic"]


def read(file):
    return json.loads(file.read_text())


def gains():
    old = read(ROOT / "scaling_siblings_20261109/potts/result_summary.json")
    decay = read(ROOT / "decay_20261114/evaluation.json")["comparisons"]
    nuclei = read(ROOT / "instances_20261114/evaluation.json")["comparisons"]
    first_run = read(ROOT / "presence_20261114/evaluation.json")["comparisons"]
    raw_only = read(ROOT / "presence_20261114/raw_only_evaluation.json")
    replication = read(HERE / "evaluation.json")["cohorts"]["primary"]["comparisons"]
    panels = [
        ("Gain over Raw TP", [
            ("Hierarchy + Potts (100)", old["delta"], old["ci95_vs_raw"]),
            ("Raw-only component WCSP (100)", raw_only["delta"], raw_only["ci95"]),
            ("Type-set WCSP, initial (100)", *unpack(first_run["vote_presence_evidence_1 minus raw_tp"])),
            ("Type-set WCSP, frozen (300)", *unpack(replication["vote_presence_evidence_1 minus raw_tp"])),
        ]),
        ("Gain over each matched control", [
            ("Block decay vs. V1 (100)", *unpack(decay["blocks_0.6 minus v1"])),
            ("Nucleus Potts vs. voting (100)", *unpack(nuclei["vote_potts_2 minus instance_vote"])),
            ("Type-set vs. voting, initial (100)", *unpack(first_run["vote_presence_evidence_1 minus vote"])),
            ("Type-set vs. voting, frozen (300)", *unpack(replication["vote_presence_evidence_1 minus vote"])),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.1, 3.8), layout="constrained")
    for ax, (title, rows) in zip(axes, panels):
        ax.axvline(0, color="#606b71", linestyle="--", lw=1)
        for i, (_, point, (lo, hi)) in enumerate(rows):
            color = "#17755f" if i >= 2 else "#4d6e99"
            ax.hlines(i, lo, hi, color=color, lw=2)
            ax.plot(point, i, "o", color=color)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.set_ylim(len(rows) - .55, -.45)
        ax.set(xlabel="Change in biological mIoU (decimal), 95% paired CI", title=title)
    fig.suptitle("PanNuke gains within each experiment  |  Sample size in parentheses", fontweight="bold")
    fig.savefig(HERE / "gains_across_experiments.png", dpi=160)
    plt.close(fig)


def unpack(row):
    return row["delta"], row["ci95"]


def audit():
    module = load_previous()
    score, confusion = module.nucleus.score, module.nucleus.confusion
    cohort = read(HERE / "split.json")["primary"]
    priors = np.array(read(ROOT / "presence_20261114/priors.json")["band_probability"])
    counts = {name: dict(errors=0, absent=0, far=0) for name in ("raw_tp", "vote", "wcsp")}
    oracle = np.zeros((6, 6), dtype=np.int64)
    matrices = {name: np.zeros_like(oracle) for name in counts}
    solve_times, candidates = [], []
    repairs = {name: dict(fixes=0, breaks=0) for name in ("within_2px", "farther")}
    for record in cohort:
        idx = record["image_index"]
        with np.load(CACHE / f"gt_{idx:04d}.npz") as f:
            gt = f["types"][f["instances"]]
            gt_ids = f["instances"]
        with np.load(CACHE / f"pred_{idx:04d}.npz") as f:
            mask, p = f["instances"], f["p"]
        vote, winner = [np.load(CACHE / "outputs" / key / f"{idx:04d}.npy")
                        for key in ("vote", "vote_presence_evidence_1")]
        raw = p.argmax(-1)
        base, refined = np.r_[0, vote][mask], np.r_[0, winner][mask]
        bound = np.zeros(gt.shape, bool)
        bound[1:] |= gt_ids[1:] != gt_ids[:-1]
        bound[:-1] |= gt_ids[1:] != gt_ids[:-1]
        bound[:, 1:] |= gt_ids[:, 1:] != gt_ids[:, :-1]
        bound[:, :-1] |= gt_ids[:, 1:] != gt_ids[:, :-1]
        near = distance_transform_edt(~bound) <= 2 if bound.any() else bound
        types = np.bincount(gt.ravel(), minlength=6) > 0
        for name, pred in zip(counts, (raw, base, refined)):
            wrong = pred != gt
            absent = wrong & (pred != 0) & ~types[pred]
            counts[name]["errors"] += int(wrong.sum())
            counts[name]["absent"] += int(absent.sum())
            counts[name]["far"] += int((wrong & ~near).sum())
            matrices[name] += confusion(gt, pred)
            if name == "vote":
                oracle += confusion(gt, np.where(absent, gt, pred))
        for name, region in (("within_2px", near), ("farther", ~near)):
            repairs[name]["fixes"] += int((region & (base != gt) & (refined == gt)).sum())
            repairs[name]["breaks"] += int((region & (base == gt) & (refined != gt)).sum())
        delta = float(score(confusion(gt, refined)) - score(confusion(gt, base)))
        if delta > 1e-12:
            candidates.append((delta, idx))
        start = time.perf_counter()
        local_vote, unary, _ = module.nucleus.unaries(mask, p)
        n = len(local_vote)
        if n:
            labels, costs = module.exact_set_options(unary["vote"])
            freq = priors[np.searchsorted(module.BANDS, n, side="right"), module.MASKS]
            evidence = module.ACTIVE @ -np.log(np.maximum(np.exp(-unary["mean"]).max(0), 1e-12))
            result = labels[(costs - np.log(freq) + evidence).argmin()]
        else:
            result = local_vote
        solve_times.append(time.perf_counter() - start)
        np.testing.assert_array_equal(result + 1, winner)
    with np.load(HERE / "primary_confusions.npz") as f:
        for check, key in zip(matrices, ("raw_tp", "vote", "vote_presence_evidence_1")):
            np.testing.assert_array_equal(matrices[check], f[key].sum(0))
    fit_counts = np.array(read(ROOT / "presence_20261114/priors.json")["set_counts"]).sum(0)
    fit_support = [int(fit_counts[np.arange(32) & (1 << k) != 0].sum()) for k in range(5)]
    example = sorted(candidates)[len(candidates) // 2]
    details = dict(n_patches=len(cohort), errors=counts, edits=repairs, vote_absent_type_oracle=float(score(oracle)),
                   fit_support=fit_support, median_cached_seconds=float(np.median(solve_times)),
                   p95_cached_seconds=float(np.percentile(solve_times, 95)), example=list(example))
    (HERE / "diagnostics.json").write_text(json.dumps(details, indent=2) + "\n")
    example_figure(example)
    print(json.dumps(details, indent=2))


def example_figure(example):
    delta, idx = example
    with np.load(CACHE / f"gt_{idx:04d}.npz") as f:
        gt = f["types"][f["instances"]]
    with np.load(CACHE / f"pred_{idx:04d}.npz") as f:
        mask = f["instances"]
    vote, winner = [np.load(CACHE / "outputs" / key / f"{idx:04d}.npy")
                    for key in ("vote", "vote_presence_evidence_1")]
    base, refined = np.r_[0, vote][mask], np.r_[0, winner][mask]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.45))
    for ax, title, im in zip(axes, ("H&E", "Ground truth", "Fixed-mask voting", "Type-set WCSP"),
                            (np.load(CACHE / f"rgb_{idx:04d}.npy"), gt, base, refined)):
        ax.imshow(im, cmap=ListedColormap(COLORS), vmin=0, vmax=5, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, top=.8, bottom=.13, wspace=.08)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)], ncol=6,
               loc="lower center", frameon=False, fontsize=9)
    fig.suptitle(f"Patch {idx} · median positive change vs. voting ({delta:+.5f} mIoU)")
    fig.savefig(HERE / "type_set_example.png", dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    gains()
    audit()
