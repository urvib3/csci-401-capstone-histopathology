"""Presence (type-set) gain over voting per organ, with 95% intervals, for Deliverable 7.

Numbers come from each organ's evaluation.json (Colon from Aakanksha's report) and the
type-set sweep; see organ_wcsp/README.md.

    .venv/bin/python experiments/pannuke_wcsp/organ_wcsp/make_organ_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WORKS = "#2a78d6"
NOT_SIGNIFICANT = "#a9a8a2"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"

# (label, gain, low, high, significant)
ROWS = [
    ("Breast, tuned for Breast\n169 test patches", 0.0141, 0.0078, 0.0213, True),
    ("Colon, tuned for Colon\n106 test patches", 0.0019, -0.0013, 0.0055, False),
    ("Adrenal, tuned for Adrenal\n144 test patches", 0.0026, -0.0052, 0.0111, False),
    ("Adrenal, original all-organ setting\n144 test patches", 0.0049, 0.0016, 0.0089, True),
    ("All organs, wider sweep\n600 fresh test patches", 0.0124, 0.0021, 0.0233, True),
]


def main():
    figure, axis = plt.subplots(figsize=(8, 3.8), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    for row_number, (label, gain, low, high, significant) in enumerate(ROWS):
        y = len(ROWS) - 1 - row_number
        color = WORKS if significant else NOT_SIGNIFICANT
        axis.plot([low, high], [y, y], color=color, linewidth=2.5, solid_capstyle="round")
        axis.plot(gain, y, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        axis.text(high + 0.0006, y, f"{gain:+.4f}", va="center", fontsize=8.5, color=TEXT_PRIMARY)
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    axis.set_yticks(range(len(ROWS)))
    labels = []
    for row in reversed(ROWS):
        labels.append(row[0])
    axis.set_yticklabels(labels, fontsize=8, color=TEXT_PRIMARY)
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8)
    axis.set_xlim(-0.008, 0.028)
    axis.set_xlabel("mIoU gain of presence over voting (95% paired bootstrap interval)", fontsize=9,
                    color=TEXT_SECONDARY)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID_COLOR)
    handles = [plt.Line2D([], [], color=WORKS, marker="o", linewidth=2.5, label="Interval above zero"),
               plt.Line2D([], [], color=NOT_SIGNIFICANT, marker="o", linewidth=2.5, label="Could be zero")]
    axis.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8,
                labelcolor=TEXT_PRIMARY)
    figure.suptitle("PanNuke presence rule: gain over voting by organ", x=0.02, ha="left", fontsize=11,
                    color=TEXT_PRIMARY, fontweight="bold")
    figure.tight_layout()
    figure.savefig(HERE / "organ_presence_gains.png", facecolor=SURFACE)
    print("Wrote", HERE / "organ_presence_gains.png")


if __name__ == "__main__":
    main()
