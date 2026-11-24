# How cluster WCSP works, on the same BCSS dev tile as the presence diagram:
# U-Net output -> SLIC patches -> one label per patch (cluster WCSP) -> pixel refine.
#
#   python deliverables/figures_20261123/cluster_diagram.py

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import COLORS  # noqa: E402
from wcsp.solve.clusters import ClusterSolver  # noqa: E402
from wcsp.viz import overlay  # noqa: E402

TILE = "TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0_512_3072_size512"
RUN = Path("/scratch1/ubhuwani/wcsp/runs/c2_both_256_refine/dev/shard_0000_of_0001.npz")
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    cfg = load_config("experiments/bcss_wcsp/configs/bcss_clusters.yaml")
    data, store = pipeline.dataset_for(cfg, "dev"), pipeline.store_for(cfg, "dev")
    sample = data[data.index(TILE)]
    problem = pipeline.problem_for(cfg, sample, store.load(TILE), 1.0)
    solver = ClusterSolver(n_segments=256, compactness=10.0, features="both", prob_weight=30.0)
    solver.prepare(sample, None)
    segments = solver.segments(problem)
    patch_labels = solver.solve(problem, problem.unary.argmin(-1)).labels
    z = np.load(RUN)
    i = [str(t) for t in z["ids"]].index(TILE)
    img, raw, final = sample.grid_image[..., :3], z["argmax"][i], z["wcsp"][i]

    edges = find_boundaries(segments, mode="inner")
    slic_img = np.asarray(img, dtype=np.float64).copy()
    slic_img[edges] = [255, 255, 255]
    patch_img = overlay(img, patch_labels).astype(np.float64)
    patch_img[edges] = [255, 255, 255]

    panels = [(overlay(img, raw), "1 · U-Net output"), (slic_img.astype(np.uint8), f"2 · SLIC: {segments.max() + 1} patches"),
              (patch_img.astype(np.uint8), "3 · One label per patch"), (overlay(img, final), "4 · Pixel refine")]
    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 0.18, 1, 0.18, 1, 0.18, 1], wspace=0.04)
    for k, (im, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * k])
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold")
        if k < 3:
            arr = fig.add_subplot(gs[0, 2 * k + 1])
            arr.axis("off")
            arr.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                         arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 2, "mutation_scale": 16})
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.asarray(COLORS[k]) / 255) for k in (0, 2, 4)]
    fig.legend(handles, ["Tumor", "Stroma", "Other"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    fig.savefig(OUT / "cluster_diagram.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
