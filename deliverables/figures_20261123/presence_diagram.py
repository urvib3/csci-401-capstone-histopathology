# How the presence rule works, on one real BCSS dev tile (tile 1 of the method examples):
# U-Net output and its class set -> presence picks an allowed set -> WCSP within that set.
#
#   python deliverables/figures_20261123/presence_diagram.py

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

TILE = "TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0_512_3072_size512"
RUN = Path("/scratch1/ubhuwani/wcsp/runs/r_g3000/dev/shard_0000_of_0001.npz")
NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
INK, MUTED, SURFACE, BOX = "#0b0b0b", "#52514e", "#fcfcfb", "#f3f2ee"


def class_set(labels, target):
    ok = target != IGNORE
    share = np.bincount(labels[ok].astype(int), minlength=5)[:5] / ok.sum()
    return [NAMES[k] for k in range(5) if share[k] >= 0.01]


def main():
    cfg = load_config("experiments/bcss_wcsp/configs/bcss_rules.yaml")
    data = pipeline.dataset_for(cfg, "dev")
    sample = data[data.index(TILE)]
    z = np.load(RUN)
    i = [str(t) for t in z["ids"]].index(TILE)
    img, target, raw, out = sample.grid_image[..., :3], sample.grid_target, z["argmax"][i], z["wcsp"][i]

    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.38, 1.35, 0.38, 1], wspace=0.05)
    panels = [(0, overlay(img, raw), "1 · U-Net output", "Classes: " + ", ".join(class_set(raw, target))),
              (4, overlay(img, out), "3 · WCSP within the kept set", "Classes: " + ", ".join(class_set(out, target)))]
    for col, im, title, sub in panels:
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold")
        ax.text(0.5, -0.07, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10, color=INK)

    mid = fig.add_subplot(gs[0, 2])
    mid.axis("off")
    mid.set_xlim(0, 1)
    mid.set_ylim(0, 1)
    mid.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.86, facecolor=BOX, edgecolor="#d9d8d2", linewidth=1.2))
    mid.text(0.5, 0.97, "2 · Presence picks the class set", ha="center", va="bottom", fontsize=11, color=INK,
             fontweight="bold")
    lines = [("Score every possible set:", True),
             ("U-Net evidence for its classes", False),
             ("+ how rare the set is in training data", False),
             ("", False),
             ("{Tumor, Stroma, Other}   rare  ✗", False),
             ("{Tumor, Stroma}           common  ✓", False),
             ("", False),
             ("Drop \"Other\": its pixels take their", False),
             ("next-best allowed class", False)]
    for j, (text, bold) in enumerate(lines):
        mid.text(0.06, 0.84 - j * 0.085, text, fontsize=9.5, color=INK if bold else MUTED,
                 fontweight="bold" if bold else "normal", va="top", family="DejaVu Sans")
    for col in (1, 3):
        ax = fig.add_subplot(gs[0, col])
        ax.axis("off")
        ax.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 2, "mutation_scale": 18})
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.asarray(COLORS[k]) / 255) for k in (0, 2, 4)]
    fig.legend(handles, ["Tumor", "Stroma", "Other"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.savefig(OUT / "presence_diagram.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
