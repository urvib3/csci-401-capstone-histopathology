# Schematic of the multi-scale WCSP's nested blocks (no data needed).
#   left   a 32 x 32 px corner of the grid with 4, 8 and 16 px block borders
#   right  the variables and links for one 16 x 16 block, level by level
#
#   python experiments/bcss_wcsp/multiscale/make_blocks_figure.py

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

INK, MUTED = "#222222", "#888888"
NEIGHBOR, LINK = "#c0392b", "#2c6fbb"
LEVEL_COLORS = ["#f2f2f2", "#dfe9f5", "#b9d1ec", "#86afdd"]   # pixels, 4, 8, 16 px


def nested_grid(ax):
    size = 32
    ax.add_patch(Rectangle((0, 0), size, size, color=LEVEL_COLORS[0], zorder=0))
    for side, lw, color in ((1, 0.25, "#cccccc"), (4, 0.9, "#7a9cc6"), (8, 1.8, "#3f6fa8"), (16, 3.2, "#1d3f6e")):
        for t in range(0, size + 1, side):
            ax.plot([t, t], [0, size], color=color, lw=lw, solid_capstyle="butt")
            ax.plot([0, size], [t, t], color=color, lw=lw, solid_capstyle="butt")
    # highlight one block per level
    for (x, y, side, alpha) in ((0, 0, 4, 0.55), (8, 0, 8, 0.4), (16, 16, 16, 0.25)):
        ax.add_patch(Rectangle((x, y), side, side, color="#e67e22", alpha=alpha, lw=0, zorder=0.5))
    ax.text(2, 2, "4×4", ha="center", va="center", fontsize=8, color=INK, weight="bold")
    ax.text(12, 4, "8 × 8", ha="center", va="center", fontsize=9, color=INK, weight="bold")
    ax.text(24, 24, "16 × 16", ha="center", va="center", fontsize=11, color=INK, weight="bold")
    ax.set_xlim(-0.5, size + 0.5)
    ax.set_ylim(size + 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Nested blocks on a 32 × 32 px corner of the 256 × 256 grid\n"
                 "every block has its own label", fontsize=10, color=INK)


def tree(ax):
    ax.set_xlim(-48, 100)
    ax.set_ylim(-4, 100)
    ax.axis("off")
    levels = [  # (name, n nodes, y)
        ("Level 3: one 16 × 16 block", 1, 88),
        ("Level 2: its four 8 × 8 blocks", 4, 64),
        ("Level 1: sixteen 4 × 4 blocks", 16, 40),
        ("Pixels: 256 (24 neighbors each, as before)", 32, 14),
    ]
    xs = []
    for (name, n, y), color in zip(levels, LEVEL_COLORS[::-1]):
        span = 80 if n > 1 else 0
        row = [50 - span / 2 + span * i / max(n - 1, 1) for i in range(n)] if n > 1 else [50]
        xs.append(row)
        size = 5 if n <= 4 else (3.6 if n == 16 else 1.6)
        for x in row:
            ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, color=color, ec="#1d3f6e", lw=0.8))
        ax.text(-2, y, name, ha="right", va="center", fontsize=9, color=INK)
    # parent-child links (blue)
    for parent_row, child_row, (_, _, py), (_, _, cy) in zip(xs[:-1], xs[1:], levels[:-1], levels[1:]):
        per = len(child_row) // len(parent_row)
        for p, x in enumerate(parent_row):
            for c in child_row[p * per:(p + 1) * per]:
                ax.plot([x, c], [py - 2.6, cy + 2.4], color=LINK, lw=0.7, alpha=0.8)
    ax.text(50, 8, "(32 of the 256 pixels drawn)", ha="center", va="top", fontsize=7.5, color=MUTED)
    # neighbor links (red) between same-level siblings
    for row, (_, n, y) in zip(xs[1:2], levels[1:2]):
        a, b = row[0], row[1]
        ax.add_patch(FancyArrowPatch((a + 1.5, y), (b - 1.5, y), arrowstyle="<->", color=NEIGHBOR,
                                     lw=1.6, mutation_scale=9))
    ax.text(-46, -2, "red ↔  neighbor blocks: same b table (scaled by block side)\n"
                   "blue —  parent–child: μ per pixel if the labels differ\n"
                   "each block also scores how well its label fits the mean U-Net probabilities inside it",
            fontsize=8.5, color=INK, va="top")
    ax.set_title("What the solver sees for one 16 × 16 block", fontsize=10, color=INK)


def main():
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.6]})
    nested_grid(left)
    tree(right)
    fig.tight_layout()
    out = Path("experiments/bcss_wcsp/figures/multiscale_blocks.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
