# Example tiles per BCSS WCSP fix, in the old report format:
#   image | ground truth | U-Net | method | changed pixels (coloured by their new class)
# Same 3 dev tiles for every method: the 3 tiles where presence + cluster gains most pixel
# accuracy over the U-Net while removing at least one class (fixed rule, chosen once).
# Presence rows also list the class sets: U-Net's, kept by presence, and true.
#
#   python deliverables/figures_20261123/method_examples.py

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import COLORS, IGNORE  # noqa: E402
from wcsp.viz import overlay  # noqa: E402

RUNS = Path("/scratch1/ubhuwani/wcsp/runs")
METHODS = [("WCSP", "r_control", False), ("WCSP + presence", "r_g3000", True),
           ("WCSP + cluster", "c2_both_256_refine", False), ("WCSP + presence + cluster", "kc_g", True)]
SHORT = ["T", "I", "S", "N", "O"]
NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(run):
    z = np.load(RUNS / run / "dev" / "shard_0000_of_0001.npz")
    present = {}
    path = RUNS / run / "dev" / "per_sample.csv"
    for r in csv.DictReader(open(path)):
        if r.get("present"):
            present[r["id"]] = int(float(r["present"]))
    return {str(i): (a, w) for i, a, w in zip(z["ids"], z["argmax"], z["wcsp"])}, present


def set_text(mask_or_bool):
    return "{" + ", ".join(SHORT[k] for k in range(5) if mask_or_bool[k]) + "}"


def classes(labels, target):
    ok = target != IGNORE
    return np.bincount(labels[ok].astype(int), minlength=5)[:5] / max(ok.sum(), 1) >= 0.01


def changed_panel(image, before, after):
    out = np.asarray(image, dtype=np.float64) * 0.55 + 255 * 0.45
    diff = before != after
    out[diff] = np.asarray(COLORS)[after[diff]]
    return out.astype(np.uint8), diff.mean()


def main():
    cfg = load_config("experiments/bcss_wcsp/configs/bcss_rules.yaml")
    data = pipeline.dataset_for(cfg, "dev")
    runs = {name: load(run) for name, run, _ in METHODS}
    best = runs["WCSP + presence + cluster"][0]
    scores = []
    for tid, (raw, out) in best.items():
        target = data[data.index(tid)].grid_target
        ok = target != IGNORE
        if ok.mean() < 0.9:
            continue
        gain = (out[ok] == target[ok]).mean() - (raw[ok] == target[ok]).mean()
        removed = (classes(raw, target) & ~classes(out, target)).any()
        scores.append((gain if removed else -1, tid))
    picks = [t for _, t in sorted(scores, reverse=True)[:3]]
    print("tiles:", picks)

    for name, run, has_presence in METHODS:
        labels, present = runs[name]
        fig, axes = plt.subplots(3, 5, figsize=(13, 8.6), gridspec_kw={"hspace": 0.42, "wspace": 0.04})
        for r, tid in enumerate(picks):
            sample = data[data.index(tid)]
            img, target = sample.grid_image[..., :3], sample.grid_target
            raw, out = labels[tid]
            ok = target != IGNORE
            acc = lambda a: 100 * (a[ok] == target[ok]).mean()  # noqa: E731
            changed, frac = changed_panel(img, raw, out)
            panels = [(img, "Image"), (overlay(img, target), "Ground truth"),
                      (overlay(img, raw), f"U-Net ({acc(raw):.1f}%)"),
                      (overlay(img, out), f"{name} ({acc(out):.1f}%)"), (changed, f"Changed ({100 * frac:.1f}%)")]
            for c, (data_, title) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(data_)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_title(title, fontsize=9.5, color=INK if c else MUTED)
            line = f"Tile {r + 1}"
            if has_presence:
                kept = [bool(present.get(tid, 0) >> k & 1) for k in range(5)]
                line += (f"  ·  classes: U-Net {set_text(classes(raw, target))} → presence keeps "
                         f"{set_text(kept)}  ·  true {set_text(classes(target, target))}")
            axes[r, 0].text(0, 1.2, line, transform=axes[r, 0].transAxes, fontsize=10, color=INK,
                            fontweight="bold", va="bottom")
        handles = [plt.Rectangle((0, 0), 1, 1, color=np.asarray(c) / 255) for c in COLORS]
        fig.legend(handles, [f"{n} ({s})" for n, s in zip(NAMES, SHORT)], loc="lower center", ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, 0.03))
        fig.suptitle(f"{name}: three dev tiles (pixel accuracy in brackets)", x=0.125, ha="left", y=0.97,
                     fontsize=12, fontweight="bold", color=INK)
        slug = name.lower().replace(" + ", "_").replace(" ", "_")
        fig.savefig(OUT / f"examples_{slug}.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
        print("wrote", slug)


if __name__ == "__main__":
    main()
