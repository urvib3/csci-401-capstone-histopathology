"""Render development and paired evaluation results from saved metrics."""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    dev = read("dev.json")
    raw = dev[0]["raw"]["biological_miou"]
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    for prefix, label in [("blocks_", "Block decay"), ("all_", "All-level decay"),
                          ("pixels_", "Pixel-only control"), ("potts_blocks_", "Potts block decay")]:
        baseline = dev[1] if prefix == "potts_blocks_" else dev[0]
        pairs = [(1.0, baseline["biological_miou"])]
        pairs += [(float(r["config"]["name"][len(prefix):]), r["biological_miou"])
                  for r in dev if r["config"]["name"].startswith(prefix)]
        x, y = zip(*sorted(pairs))
        ax.plot(x, [(v - raw)*100 for v in y], marker="o", label=label)
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(xlabel="Decay factor (1 = equal weights)", ylabel="Biological mIoU change vs. Raw TP (pp)",
           title="PanNuke development sweep · 12 patches")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(HERE / "development.png", dpi=165)
    plt.close(fig)
    with (HERE / "development.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", *[f"beta_{i}" for i in range(1, 7)], "gamma", "bio_miou",
                         "delta_raw_pp", "fixes", "breaks", "wrong_to_different_wrong", "seconds"])
        for row in dev:
            config = row["config"]
            writer.writerow([config["name"], *config["beta"], config["gamma"], row["biological_miou"],
                             (row["biological_miou"] - raw)*100, row["fixes"], row["breaks"],
                             row["different_wrong"], row["seconds"]])
    if not (HERE / "evaluation.json").exists():
        return
    final = read("evaluation.json")
    labels = {"v1": "Original V1", "potts": "Original Potts", "matched_potts": "Potts with matched settings"}
    frozen = read("frozen.json")
    labels.setdefault(frozen["hybrid"]["name"], "V2 · equal weights")
    labels.setdefault(frozen["potts"]["name"], "Dev-selected Potts")
    diagnostic = frozen["best_nonuniform_hybrid_decay"]["name"]
    labels.setdefault(diagnostic, "V1 · block decay (d = 0.60)")
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    for i, name in enumerate(final["results"]):
        pair = final["comparisons"][f"{name} minus raw"]
        value, (lo, hi) = pair["delta"], pair["ci95"]
        ax.hlines(i, lo*100, hi*100, color="#174b78", lw=2)
        ax.plot(value*100, i, "o", color="#174b78")
    ax.set_yticks(range(len(final["results"])), [labels.get(n, "Confidence-gated " + n.split("_gate_")[0]) for n in final["results"]])
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", ls="--", lw=.8)
    ax.set(xlabel="Biological mIoU change vs. Raw TP (pp), 95% paired CI",
           title="PanNuke comparison · 100 newly sampled patches")
    fig.savefig(HERE / "evaluation.png", dpi=165)
    plt.close(fig)

    chosen = diagnostic
    if chosen in {"v1", "potts"}:
        return
    control = "v1"
    with np.load(HERE / "confusions.npz") as archive:
        matrices = {key: archive[key] for key in ("raw", "v1", "potts", chosen)}
    scores = {}
    for name, cms in matrices.items():
        tp = np.diagonal(cms, axis1=1, axis2=2)
        union = cms.sum(1) + cms.sum(2) - tp
        scores[name] = np.divide(tp, union, out=np.zeros_like(tp, dtype=float),
                                 where=union > 0)[:, 1:].mean(1)
    diff = scores[chosen] - scores[control]
    order = np.argsort(diff, kind="stable")
    records = read("split.json")["test"]
    fig, axes = plt.subplots(3, 6, figsize=(12, 7.3))
    fig.subplots_adjust(left=.045, right=.985, top=.87, bottom=.10, hspace=.36, wspace=.13)
    palette = np.array([[24,24,24], [229,66,77], [60,160,233], [62,176,118],
                        [245,186,67], [187,118,216]], dtype=np.uint8)
    cache = HERE / "cache"
    for row, (j, title) in enumerate(zip([order[-1], order[0], order[len(order)//2]],
                                         ["Largest gain", "Largest loss", "Median position"])):
        index = records[j]["image_index"]
        rgb = np.load(cache / f"rgb_{index:04d}.npy")
        gt = np.load(cache / f"gt_{index:04d}.npy")
        with np.load(cache / f"prob_{index:04d}.npz") as file:
            raw = file["p"].argmax(-1)
        arrays = [rgb, palette[gt], palette[raw]]
        for name in ("v1", chosen, "potts"):
            with np.load(HERE / "predictions" / name / f"{index:04d}.npz") as file:
                arrays.append(palette[file["levels"][:gt.size].reshape(gt.shape)])
        titles = [f"{title}\nH&E · ID {index}", "Ground truth", "Raw TP", "V1", "Block decay", "Original Potts"]
        for col, (ax, pixels, label) in enumerate(zip(axes[row], arrays, titles)):
            ax.imshow(pixels)
            ax.set_axis_off()
            suffix = "" if col < 2 else f"\nBio mIoU {scores[('raw', 'v1', chosen, 'potts')[col-2]][j]:.4f}"
            ax.set_title(label + suffix, fontsize=8)
    class_names = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Non-neoplastic"]
    handles = [Patch(facecolor=color / 255, label=name) for color, name in zip(palette, class_names)]
    fig.legend(handles=handles, ncol=6, frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(.5, .025))
    fig.suptitle(f"Patch examples · {labels[chosen]} versus {labels[control]}")
    fig.savefig(HERE / "examples.png", dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()
