# One BCSS dev tile per failure mode of Ryan's full-res U-Net (185 frozen dev inputs),
# shown as image | ground truth | U-Net prediction:
#   grouping   most predicted pieces per true region while the majority class is right
#   labeling   largest true region (>= 20% of the tile) whose majority prediction is wrong
#   extra      most predicted classes (>= 1% of the tile) that are absent from the truth
#
#   python deliverables/figures_20261123/issue_examples.py

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

OUT = Path(__file__).resolve().parent
INPUTS = Path("data/fullres_multiscale_20261115/inputs")
IMAGES = Path("/scratch1/ubhuwani/data/bcss_v2/BCSS_512")
NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
TISSUE = ["#d64541", "#2f9e44", "#5c7cfa", "#f2c037", "#c56cf0"]
IGNORE, K = 255, 5
SURFACE, INK = "#fcfcfb", "#0b0b0b"


def image_for(tile_id):
    for split in ("train_512", "val_512"):
        p = IMAGES / split / f"{tile_id}.png"
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))
    return None


def scores(path):
    z = np.load(path)
    target, pred = z["target"], z["probs"].astype(np.float32).argmax(0)
    valid = target != IGNORE
    if valid.mean() < 0.9:
        return None
    out = {"id": str(z["id"]), "target": target, "pred": pred}
    n_true = n_pred_in = 0
    big_wrong = 0
    for c in range(K):
        comp, n = ndimage.label((target == c) & valid, structure=np.ones((3, 3)))
        for r in range(1, n + 1):
            region = comp == r
            if region.sum() < 2000:
                continue
            maj = np.bincount(pred[region], minlength=K).argmax()
            if maj == c:
                n_true += 1
                n_pred_in += len(np.unique(ndimage.label(pred == c, structure=np.ones((3, 3)))[0][region])) - 1
            elif region.mean() >= 0.2:
                big_wrong = max(big_wrong, region.mean())
    acc = (pred[valid] == target[valid]).mean()
    out["grouping"] = n_pred_in / max(n_true, 1) if acc > 0.7 else 0
    out["labeling"] = big_wrong
    share = lambda a: np.bincount(a[valid], minlength=K)[:K] / valid.sum() >= 0.01  # noqa: E731
    out["extra"] = int((share(pred) & ~share(target)).sum()) + 0.001 * acc
    return out


def main():
    rows = [r for r in (scores(p) for p in sorted(INPUTS.glob("*.npz"))) if r is not None]
    picks = [("Segmentation: one true region broken into pieces", max(rows, key=lambda r: r["grouping"])),
             ("Class identification: a whole region given the wrong class", max(rows, key=lambda r: r["labeling"])),
             ("Too many tissue types: classes not in the tile", max(rows, key=lambda r: r["extra"]))]
    cmap = ListedColormap(TISSUE + ["#e4e3df"])
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 10.2), gridspec_kw={"hspace": 0.22, "wspace": 0.04})
    for i, (title, r) in enumerate(picks):
        img = image_for(r["id"])
        show = lambda a: np.where(r["target"] == IGNORE, K, a)  # noqa: E731
        panels = [(img, "Image"), (show(r["target"]), "Ground truth"), (show(r["pred"]), "U-Net")]
        for j, (data, name) in enumerate(panels):
            ax = axes[i, j]
            if data is None:
                ax.text(0.5, 0.5, "image not found", ha="center", va="center")
            elif data.ndim == 3:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=K, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(name, color=INK, fontsize=11)
        axes[i, 0].text(0, 1.04, title, transform=axes[i, 0].transAxes, color=INK, fontsize=11,
                        fontweight="bold", va="bottom")
        print(title, r["id"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TISSUE] + [plt.Rectangle((0, 0), 1, 1, color="#e4e3df")]
    fig.legend(handles, NAMES + ["Unlabeled"], loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.04))
    fig.savefig(OUT / "issue_examples.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
