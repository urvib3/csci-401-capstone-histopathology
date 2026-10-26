# Figures for the BCSS WCSP write-up:
#   pipeline.png  one val tile through every step (512 tile -> U-Net -> upsample -> crop -> WCSP)
#   wcsp_diagram.png  the 24-pixel neighborhood with its 1/d weights, and the block scheme
#
#   python experiments/bcss_wcsp/make_figures.py --id <val tile id>

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wcsp import pipeline  # noqa: E402
from wcsp.classifiers.base import Prediction  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import BCSS_TO_WCSP, COLORS, IGNORE, reorder_probs  # noqa: E402
from wcsp.model.neighborhood import distance_weights, forward_offsets  # noqa: E402
from wcsp.solve.argmax import ArgmaxSolver  # noqa: E402
from wcsp.solve.blocks import BlockLayout  # noqa: E402
from wcsp.solve.factory import get_solver  # noqa: E402
from wcsp.viz import overlay  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]   # wcsp label order


def run_unet(sample):
    """Raw half-resolution softmax over the whole tile, and the x2-upsampled version."""
    import torch
    import torch.nn.functional as F
    from tiatoolbox.models.architecture import get_pretrained_model

    model, _ = get_pretrained_model("fcn_resnet50_unet-bcss")
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(sample.image[None])).permute(0, 3, 1, 2).float()
    with torch.inference_mode():
        half = F.softmax(model(model.preproc(x)), dim=1)
        full = F.interpolate(half, scale_factor=2, mode="bilinear", align_corners=False)
    half = reorder_probs(half[0].numpy(), BCSS_TO_WCSP, axis=0)
    full = reorder_probs(full[0].numpy(), BCSS_TO_WCSP, axis=0)
    return half, full


def accuracy(labels, target):
    scored = target != IGNORE
    return (labels[scored] == target[scored]).mean()


def box(ax, size, lo, hi):
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo, fill=False, ec="white", lw=2, ls="--"))
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)


def legend(fig):
    from matplotlib.patches import Patch

    handles = [Patch(color=c / 255, label=n) for n, c in zip(NAMES, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(NAMES), fontsize=10, frameon=False)


def pipeline_figure(cfg, sample_id, out):
    import matplotlib.pyplot as plt

    data = pipeline.dataset_for(cfg, "val")
    sample = data[data.index(sample_id)]
    half, full = run_unet(sample)
    crop = full[:, 128:384, 128:384]

    problem = pipeline.problem_for(cfg, sample, Prediction(np.ascontiguousarray(crop)), 1.0)
    init = ArgmaxSolver().solve(problem, np.zeros(problem.shape, dtype=np.int64)).labels
    refined = get_solver(cfg.solver).solve(problem, init).labels

    image, target = sample.image, sample.grid_target
    small = image[::2, ::2]
    panels = [
        ("1. Input tile\n512 × 512", image, 512),
        ("2. U-Net output\n256 × 256 (half res, whole tile)", overlay(small, half.argmax(0)), 256),
        ("3. Upsample probabilities ×2\n512 × 512, class = argmax", overlay(image, full.argmax(0)), 512),
        (f"4. Keep central 256 × 256\nU-Net labels (acc {accuracy(init, target):.1%})", overlay(sample.grid_image, init), None),
        (f"5. WCSP on the crop\n256 × 256 (acc {accuracy(refined, target):.1%})", overlay(sample.grid_image, refined), None),
        ("Ground truth\n(central 256 × 256)", overlay(sample.grid_image, target), None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 4.2))
    for ax, (title, img, size) in zip(axes, panels):
        ax.imshow(img)
        if size:
            box(ax, size, size // 4, 3 * size // 4)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "pipeline.png", dpi=110)
    plt.close(fig)
    print(f"wrote {out / 'pipeline.png'}: U-Net acc {accuracy(init, target):.3f}, WCSP acc {accuracy(refined, target):.3f}")


def diagram(cfg, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

    # Neighborhood: every pixel pair up to 2√2 apart, weighted 1/d (then normalized).
    offsets = forward_offsets(cfg.problem.types)
    weights = distance_weights(offsets, cfg.problem.eta, normalize=True)
    grid = np.full((5, 5), np.nan)
    for (dy, dx), w in zip(offsets, weights):
        grid[2 + dy, 2 + dx] = grid[2 - dy, 2 - dx] = w
    left.imshow(grid, cmap="Blues", vmin=0, vmax=np.nanmax(grid) * 1.3)
    for y in range(5):
        for x in range(5):
            d = np.hypot(y - 2, x - 2)
            text = "center\npixel" if d == 0 else f"d={d:.2g}\nw={grid[y, x]:.3f}"
            left.text(x, y, text, ha="center", va="center", fontsize=7.5)
    left.set_title("Pairwise neighborhood (24 pixels)\ncost = λ · w · b[label, label'],  w ∝ 1/d,  Σw = 1", fontsize=10)
    left.set_xticks([])
    left.set_yticks([])

    # Block scheme on the 256 x 256 crop.
    right.add_patch(Rectangle((0, 0), 256, 256, color="#f2c14e", alpha=0.6))
    for name, (x, y) in zip("ABCD", [(0, 0), (156, 0), (0, 156), (156, 156)]):
        right.add_patch(Rectangle((x, y), 100, 100, color="#4c78a8", alpha=0.75))
        right.text(x + 50, y + 50, name, color="white", ha="center", va="center", fontsize=16, weight="bold")
    right.text(128, 128, "cross\n56 px", ha="center", va="center", fontsize=10)
    right.set_xlim(0, 256)
    right.set_ylim(256, 0)
    right.set_aspect("equal")
    right.set_xticks([0, 100, 156, 256])
    right.set_yticks([0, 100, 156, 256])
    right.set_title("Block scheme on the 256 × 256 crop\n"
                    "① fix cross, solve A–D (independent)  ② fix A–D, solve cross  → repeat", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "wcsp_diagram.png", dpi=110)
    plt.close(fig)
    print(f"wrote {out / 'wcsp_diagram.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--id", required=True)
    parser.add_argument("--lam", default="4")
    parser.add_argument("--out", default="experiments/bcss_wcsp/figures")
    args = parser.parse_args()
    cfg = load_config(args.config, [f"problem.lam={args.lam}"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diagram(cfg, out)
    pipeline_figure(cfg, args.id, out)


if __name__ == "__main__":
    main()
