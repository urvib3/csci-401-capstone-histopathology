"""Summary figure for one organ: each family's gain over its matched control, organ-tuned vs. global.

Usage (from the repository root):
    .venv/bin/python experiments/pannuke_wcsp/organ_wcsp/make_summary.py --organ Adrenal_gland
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

FAMILY_LABELS = {
    "pixel_table": "Pixel learned table",
    "hierarchy": "Hierarchy V1/V2",
    "scaling_siblings": "Scaling + siblings",
    "decay": "Decay schedules",
    "nucleus_pairwise": "Nucleus pairwise",
    "type_counts": "Type counts",
    "type_set": "Type set",
    "raw_component": "Raw TP component",
}
CONTROL_LABELS = {"raw_tp": "Raw TP", "fixed_mask_vote": "fixed-mask vote", "organ_v1": "V1", "organ_v2": "V2"}
ORGAN_COLOR = "#2a78d6"
GLOBAL_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#55554f"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def main(organ):
    out = HERE / organ.lower()
    evaluation = json.loads((out / "evaluation.json").read_text())
    families = list(FAMILY_LABELS)

    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)

    for row_number, family in enumerate(families):
        result = evaluation["families"][family]
        y = len(families) - 1 - row_number
        series = [
            (result["organ_vs_control"], ORGAN_COLOR, "o", y + 0.14),
            (result["global_vs_control"], GLOBAL_COLOR, "s", y - 0.14),
        ]
        for comparison, color, marker, position in series:
            low, high = comparison["ci95"]
            axis.plot([low, high], [position, position], color=color, linewidth=2, solid_capstyle="round")
            axis.plot(comparison["delta"], position, marker=marker, markersize=8, color=color,
                      markeredgecolor=SURFACE, markeredgewidth=1.5)

    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    axis.set_yticks(range(len(families)))
    labels = []
    for family in reversed(families):
        control = CONTROL_LABELS[evaluation["families"][family]["control"]]
        labels.append(f"{FAMILY_LABELS[family]}\nvs. {control}")
    axis.set_yticklabels(labels, fontsize=8.5, color=TEXT_PRIMARY)
    axis.set_xlabel("Change in pooled biological mIoU vs. matched control (95% paired bootstrap CI)",
                    fontsize=9, color=TEXT_SECONDARY)
    axis.tick_params(axis="x", labelsize=8, colors=TEXT_SECONDARY)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID_COLOR)

    organ_handle = plt.Line2D([], [], color=ORGAN_COLOR, marker="o", markersize=7, linewidth=2,
                              label="Tuned for this organ (fit + dev on this organ)")
    global_handle = plt.Line2D([], [], color=GLOBAL_COLOR, marker="s", markersize=7, linewidth=2,
                               label="Original global settings, unchanged")
    axis.legend(handles=[organ_handle, global_handle], loc="upper center", bbox_to_anchor=(0.5, 1.13),
                ncol=2, frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY)
    title = f'{organ.replace("_", " ")}: {evaluation["n_test_patches"]} test patches'
    figure.suptitle(title, x=0.02, ha="left", fontsize=11, color=TEXT_PRIMARY, fontweight="bold")
    figure.tight_layout()
    figure.savefig(out / "summary.png", facecolor=SURFACE)
    print("Wrote", out / "summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organ", required=True)
    main(parser.parse_args().organ)
