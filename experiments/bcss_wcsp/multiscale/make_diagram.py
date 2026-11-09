# Figure for the multi-scale block WCSP (wcsp/solve/hierarchy.py):
#   top row     how the levels connect
#   bottom row  one dev tile: U-Net labels, the block labels at each level, the
#               final pixel labels and the ground truth
#
#   python experiments/bcss_wcsp/multiscale/make_diagram.py --id <dev tile id> --mu 0.2

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import COLORS  # noqa: E402
from wcsp.solve.argmax import ArgmaxSolver  # noqa: E402
from wcsp.solve.factory import get_solver  # noqa: E402
from wcsp.viz import overlay  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]


def schematic(ax):
    from matplotlib.patches import FancyArrowPatch, Rectangle

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")
    rows = [  # (label, cells per side, y)
        ("Level 3: 16 × 16 px blocks (256 px)", 2, 34),
        ("Level 2: 8 × 8 px blocks (64 px)", 4, 23),
        ("Level 1: 4 × 4 px blocks (16 px)", 8, 12),
        ("Level 0: pixels (24 neighbors, as before)", 32, 1),
    ]
    width, height, x0 = 40, 8, 30
    for name, n, y in rows:
        cell = width / n
        for i in range(n):
            ax.add_patch(Rectangle((x0 + i * cell, y), cell, height, fill=False, lw=0.4 if n > 8 else 1.2,
                                   ec="#444"))
        ax.text(x0 - 1.5, y + height / 2, name, ha="right", va="center", fontsize=9)
    # neighbor links (same level) and parent-child links (between levels)
    for (_, n, y) in rows[:3]:
        cell = width / n
        ax.add_patch(FancyArrowPatch((x0 + cell / 2, y + height / 2), (x0 + 1.5 * cell, y + height / 2),
                                     arrowstyle="<->", color="#d62728", lw=1.6, mutation_scale=8))
    for (_, n, y), (_, m, y2) in zip(rows[:3], rows[1:]):
        cell, child = width / n, width / m
        for c in range(m // n):
            ax.plot([x0 + cell / 2, x0 + (c + 0.5) * child], [y, y2 + height], color="#1f77b4", lw=1)
    ax.text(x0 + width + 2, 30, "red ↔ : neighbor blocks, cost  block_lam · side · w · b[label, label′]",
            fontsize=8.5, color="#d62728", va="center")
    ax.text(x0 + width + 2, 25, "blue — : parent–child, cost  μ · child area  if labels differ",
            fontsize=8.5, color="#1f77b4", va="center")
    ax.text(x0 + width + 2, 20, "each block also has a unary:  area · −log(mean U-Net p in the block)",
            fontsize=8.5, va="center")
    ax.text(x0 + width + 2, 13, "solve: fix all levels but one, solve it, cycle\npixels → level 1 → 2 → 3, repeat",
            fontsize=8.5, va="center")


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_multiscale.yaml")
    parser.add_argument("--id", required=True)
    parser.add_argument("--mu", default="0.2")
    parser.add_argument("--levels", default="3")
    parser.add_argument("--out", default="experiments/bcss_wcsp/figures/multiscale_diagram.png")
    args = parser.parse_args()
    cfg = load_config(args.config, [f"solver.mu={args.mu}", f"solver.levels={args.levels}"])

    data = pipeline.dataset_for(cfg, "dev")
    sample = data[data.index(args.id)]
    prediction = pipeline.store_for(cfg, "dev").load(sample.id)
    problem = pipeline.problem_for(cfg, sample, prediction, 1.0)
    init = ArgmaxSolver().solve(problem, np.zeros(problem.shape, dtype=np.int64)).labels
    solver = get_solver(cfg.solver)
    result = solver.solve(problem, init)

    image = sample.grid_image
    panels = [("U-Net labels\n(pixels)", init, 1)]
    for side, z in reversed(list(zip(solver.sides(), result.levels))):
        panels.append((f"Level {solver.sides().index(side) + 1} block labels\n{side} × {side} px", z, side))
    panels += [(f"Multi-scale WCSP\n(pixels, μ = {args.mu})", result.labels, 1),
               ("Ground truth", sample.grid_target, 1)]

    fig = plt.figure(figsize=(18, 8.4))
    top = fig.add_axes((0.02, 0.56, 0.96, 0.42))
    schematic(top)
    width = 0.96 / len(panels)
    for i, (title, labels, side) in enumerate(panels):
        ax = fig.add_axes((0.02 + i * width, 0.07, width * 0.94, 0.42))
        full = np.repeat(np.repeat(labels, side, 0), side, 1)
        ax.imshow(overlay(image, full))
        if side >= 8:
            for t in range(0, image.shape[0] + 1, side):
                ax.axhline(t - 0.5, color="white", lw=0.3, alpha=0.6)
                ax.axvline(t - 0.5, color="white", lw=0.3, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    handles = [Patch(color=c / 255, label=n) for n, c in zip(NAMES, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(NAMES), fontsize=10, frameon=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
