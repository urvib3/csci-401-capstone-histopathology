"""Gain over voting vs. strength, with the winner's other knobs fixed, on dev and test.

    .venv/bin/python experiments/pannuke_wcsp/type_set_sweep_20261123/make_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEV_COLOR = "#2a78d6"
TEST_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#55554f"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def main():
    development = json.loads((HERE / "development.json").read_text())
    evaluation = json.loads((HERE / "evaluation.json").read_text())
    frozen = json.loads((HERE / "frozen.json").read_text())
    winner = frozen["winner"]

    dev_vote = development["controls"]["fixed_mask_vote"]
    dev_points = {}
    for row in development["rows"]:
        config = row["config"]
        same_other_knobs = True
        for knob in ("mode", "unary", "bands", "smoothing", "organ_specific"):
            if config[knob] != winner[knob]:
                same_other_knobs = False
        if same_other_knobs:
            dev_points[config["strength"]] = 100 * (row["dev_bio_miou"] - dev_vote)

    strengths = sorted(dev_points)
    dev_gains = [dev_points[s] for s in strengths]
    test_curve = evaluation["test_strength_curve_exploratory"]
    test_gains = [100 * test_curve[str(s)]["gain_over_vote"] for s in strengths]
    original_gain = 100 * evaluation["exploratory"]["original_setting_original_counts vs fixed_mask_vote"]["delta"]

    figure, axis = plt.subplots(figsize=(8, 4.6), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    axis.plot(strengths, dev_gains, color=DEV_COLOR, linewidth=2, marker="o", markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=1.5, label="Knob-picking images (400)")
    axis.plot(strengths, test_gains, color=TEST_COLOR, linewidth=2, marker="s", markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=1.5, label="Final-score images (600, checked after freezing)")
    axis.axhline(original_gain, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    axis.text(5, original_gain + 0.08, "Original setting, final-score images", fontsize=8,
              color=TEXT_SECONDARY)
    axis.axhline(0, color=GRID_COLOR, linewidth=1)
    axis.axvline(winner["strength"], color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    axis.text(winner["strength"] * 1.05, 0.08, f'Chosen on knob-picking images: {winner["strength"]:g}', fontsize=8,
              color=TEXT_SECONDARY)
    axis.axvspan(strengths[0] * 0.9, 4, color=GRID_COLOR, alpha=0.5, linewidth=0)
    axis.text(0.27, 0.08, "Range tried before this sweep", fontsize=8, color=TEXT_SECONDARY)

    axis.set_xscale("log", base=2)
    axis.set_xticks(strengths)
    axis.set_xticklabels([f"{s:g}" for s in strengths])
    axis.set_xlabel("Strength of the type-set cost", fontsize=9, color=TEXT_SECONDARY)
    axis.set_ylabel("Gain over voting (mIoU points)", fontsize=9, color=TEXT_SECONDARY)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID_COLOR)
    axis.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY)
    figure.suptitle("Type-set WCSP: gain over voting by strength", x=0.02, ha="left", fontsize=11,
                    color=TEXT_PRIMARY, fontweight="bold")
    figure.tight_layout()
    figure.savefig(HERE / "strength_curve.png", facecolor=SURFACE)
    print("Wrote", HERE / "strength_curve.png")


if __name__ == "__main__":
    main()
