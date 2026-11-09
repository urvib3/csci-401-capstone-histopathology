# Old WCSP vs. multi-scale WCSP vs. ground truth on the tiles where they differ most.
#
# Rows: tiles where the new WCSP helps most, a typical tile where it helps, and where it
# hurts most (pixel accuracy on labeled pixels, new minus old). Columns: image, ground
# truth, U-Net, original WCSP and its change map vs. the U-Net, multi-scale WCSP and its
# change map (green: fixed, red: broke, gray: changed on unlabeled or still-wrong pixels).
#
#   python experiments/bcss_wcsp/multiscale/compare_figure.py --old data_lam8 --new msd_mu0.2 --split dev

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import COLORS, IGNORE  # noqa: E402
from wcsp.viz import overlay  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]


def load_run(root, run, split):
    out = {}
    for shard in sorted((root / run / split).glob("shard_*.npz")):
        z = np.load(shard)
        for sample_id, pred, refined in zip(z["ids"].tolist(), z["argmax"], z["wcsp"]):
            out[sample_id] = (pred, refined)
    return out


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--old", default="data_lam8")
    parser.add_argument("--new", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--helps", type=int, default=2)
    parser.add_argument("--hurts", type=int, default=1)
    parser.add_argument("--title", default="")
    parser.add_argument("--out", default="experiments/bcss_wcsp/figures/multiscale_compare.png")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = pipeline.dataset_for(cfg, args.split)
    old = load_run(cfg.runs_root, args.old, args.split)
    new = load_run(cfg.runs_root, args.new, args.split)

    gains = []
    for sample_id in sorted(set(old) & set(new)):
        target = data[data.index(sample_id)].grid_target
        labeled = target != IGNORE
        if labeled.sum() < 2000:
            continue
        old_right = (old[sample_id][1] == target) & labeled
        new_right = (new[sample_id][1] == target) & labeled
        gains.append(((new_right.sum() - old_right.sum()) / labeled.sum(), sample_id))
    gains.sort()
    middle = [g for g in gains if g[0] > 0]
    chosen = gains[::-1][:args.helps] + ([middle[len(middle) // 2]] if middle else []) + gains[:args.hurts]
    tags = ["helps most"] * args.helps + (["typical gain"] if middle else []) + ["hurts most"] * args.hurts

    def change_map(shape, before, after, target, labeled):
        out = np.full(shape, 255, dtype=np.uint8)
        moved = before != after
        out[moved] = (175, 175, 175)
        out[moved & labeled & (after == target)] = (30, 160, 60)
        out[moved & labeled & (before == target)] = (210, 40, 40)
        return out

    columns = ["Image", "Ground truth", "U-Net", "Original WCSP", "Original vs. U-Net",
               "Multi-scale WCSP", "Multi-scale vs. U-Net"]
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(2.6 * len(columns), 2.85 * len(chosen) + 1.0),
                             squeeze=False)
    for row, ((gain, sample_id), tag) in enumerate(zip(chosen, tags)):
        sample = data[data.index(sample_id)]
        image, target = sample.grid_image, sample.grid_target
        unet, old_labels = old[sample_id]
        new_labels = new[sample_id][1]
        labeled = target != IGNORE
        acc = lambda labels: (labels[labeled] == target[labeled]).mean()  # noqa: E731
        def counts(after):  # noqa: E306
            fixed = (labeled & (unet != target) & (after == target)).sum()
            broke = (labeled & (unet == target) & (after != target)).sum()
            return f"+{fixed} / −{broke} px"
        panels = [image, overlay(image, target), overlay(image, unet), overlay(image, old_labels),
                  change_map(image.shape, unet, old_labels, target, labeled),
                  overlay(image, new_labels), change_map(image.shape, unet, new_labels, target, labeled)]
        notes = ["", "(no tint = unlabeled)", f"acc {acc(unet):.1%}", f"acc {acc(old_labels):.1%}", counts(old_labels),
                 f"acc {acc(new_labels):.1%}", counts(new_labels)]
        for col, (ax, img) in enumerate(zip(axes[row], panels)):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=10)
            ax.set_xlabel(notes[col], fontsize=8.5)
        axes[row][0].set_ylabel(tag, fontsize=10)
    handles = [Patch(color=c / 255, label=n) for n, c in zip(NAMES, COLORS)]
    handles += [Patch(color=(30 / 255, 160 / 255, 60 / 255), label="fixed (was wrong, now right)"),
                Patch(color=(210 / 255, 40 / 255, 40 / 255), label="broke (was right, now wrong)"),
                Patch(color=(175 / 255, 175 / 255, 175 / 255), label="changed, unlabeled or still wrong")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, frameon=False)
    if args.title:
        fig.suptitle(args.title, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97 if args.title else 1))
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}: {[(s[:40], round(g, 4)) for g, s in chosen]}")


if __name__ == "__main__":
    main()
