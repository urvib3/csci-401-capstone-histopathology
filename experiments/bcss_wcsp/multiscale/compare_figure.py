# Old WCSP vs. multi-scale WCSP vs. ground truth on the tiles where they differ most.
#
# Rows: tiles where the new WCSP helps most, then where it hurts most (pixel accuracy
# on labeled pixels, new minus old). Columns: image, U-Net, old WCSP, new WCSP,
# ground truth, and a difference map (green: only new is right, red: only old is right).
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
    chosen = [g for g in gains[::-1][:args.helps]] + [g for g in gains[:args.hurts]]

    columns = ["Image", "U-Net", "Old WCSP\n(pixel only)", "New WCSP\n(multi-scale)", "Ground truth",
               "New vs. old\ngreen fixed · red broke"]
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(2.9 * len(columns), 3.0 * len(chosen) + 0.9),
                             squeeze=False)
    for row, (gain, sample_id) in enumerate(chosen):
        sample = data[data.index(sample_id)]
        image, target = sample.grid_image, sample.grid_target
        unet, old_labels = old[sample_id]
        new_labels = new[sample_id][1]
        labeled = target != IGNORE
        old_right, new_right = old_labels == target, new_labels == target
        diff = np.full(image.shape, 255, dtype=np.uint8)
        diff[(new_labels != old_labels)] = (170, 170, 170)
        diff[labeled & new_right & ~old_right] = (30, 160, 60)
        diff[labeled & old_right & ~new_right] = (210, 40, 40)
        acc = lambda labels: (labels[labeled] == target[labeled]).mean()  # noqa: E731
        panels = [image, overlay(image, unet), overlay(image, old_labels), overlay(image, new_labels),
                  overlay(image, target), diff]
        notes = ["", f"acc {acc(unet):.1%}", f"acc {acc(old_labels):.1%}", f"acc {acc(new_labels):.1%}", "",
                 f"{(new_labels != old_labels).mean():.1%} px differ"]
        for col, (ax, img) in enumerate(zip(axes[row], panels)):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=10)
            ax.set_xlabel(notes[col], fontsize=9)
        axes[row][0].set_ylabel("helps most" if gain > 0 else "hurts most", fontsize=10)
    handles = [Patch(color=c / 255, label=n) for n, c in zip(NAMES, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(NAMES), fontsize=10, frameon=False)
    if args.title:
        fig.suptitle(args.title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97 if args.title else 1))
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}: {[(s[:40], round(g, 4)) for g, s in chosen]}")


if __name__ == "__main__":
    main()
