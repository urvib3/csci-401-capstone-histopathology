# Classical unsupervised clustering / superpixel methods on BCSS_512.
#
#   .venv/bin/python benchmarks/bcss/scripts/cluster_classical.py
#
# Part 1 - over-segmentation quality (val sample, native 512 px):
#   SLIC (Lab and HED), Felzenszwalb, compact watershed, quickshift,
#   mean shift, pixel k-means / GMM + connected components, RAG greedy merge,
#   normalized cut. Each method is swept over its size parameter so methods
#   can be compared at similar segment counts. Scores (common.segment_scores):
#   segments per tile, ASA (best achievable pixel accuracy if each segment
#   gets one label), oracle mIoU, boundary recall, seconds per tile.
#
# Part 2 - unsupervised classification (full val, 256 px):
#   pixel k-means / GMM fit on unlabeled train pixels; each cluster is then
#   named with the majority class of its train pixels. Scored with mIoU on the
#   held-out-hospital val split, with and without Macenko normalization.
#
# Outputs in benchmarks/bcss/results/:
#   clustering_superpixels.csv, clustering_semantic.csv,
#   figures/clustering_examples.png, figures/asa_vs_segments.png

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from common import (CLASS_NAMES, IGNORE_INDEX, LUT, NUM_CLASSES, OUTPUT_DIR, REPO, RESULTS_DIR,
                    BCSSDataset, class_report, confusion_matrix, dev_split,
                    hed_features, load_split, macenko_normalize, sample_tiles,
                    segment_scores)

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TILE_CACHE = os.path.join(OUTPUT_DIR, "classical_tiles")


# Pixel features shared by k-means / GMM: HED optical density plus a smoothed
# copy for local context (6 dims).
def pixel_features(image, sigma=4):
    from scipy.ndimage import gaussian_filter

    hed = hed_features(image)
    smooth = np.stack([gaussian_filter(hed[..., c], sigma) for c in range(3)], -1)
    return np.concatenate([hed, smooth], -1)


def fit_pixel_models(normalize, ks=(5, 8, 16), n_pixels=300_000, seed=0):
    """k-means and GMM on random pixels from the train fit tiles (labels unused)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.mixture import GaussianMixture

    images, masks, names = load_split("train", 256)
    fit = np.flatnonzero(~dev_split(names))
    rng = np.random.default_rng(seed)
    tiles = rng.choice(fit, 600, replace=False)
    per_tile = n_pixels // len(tiles)
    feats = []
    for t in tiles:
        image = macenko_normalize(images[t]) if normalize else images[t]
        f = pixel_features(image).reshape(-1, 6)
        feats.append(f[rng.choice(len(f), per_tile, replace=False)])
    x = np.concatenate(feats)
    models = {}
    for k in ks:
        models[("kmeans", k)] = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit(x)
        models[("gmm", k)] = GaussianMixture(k, covariance_type="full", random_state=seed,
                                             max_iter=200).fit(x)
    return models


def soft_assign(model, feats):
    """(H, W, k) cluster responsibilities; k-means uses a softmin over distances."""
    h, w, d = feats.shape
    flat = feats.reshape(-1, d)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(flat)
    else:
        dist = model.transform(flat) ** 2
        p = np.exp(-(dist - dist.min(1, keepdims=True)) / 0.01)
        p /= p.sum(1, keepdims=True)
    return p.reshape(h, w, -1)


def cluster_map(model, image, smooth=3):
    """Hard cluster labels after spatially smoothing the responsibilities."""
    from scipy.ndimage import gaussian_filter

    p = soft_assign(model, pixel_features(image))
    if smooth:
        p = np.stack([gaussian_filter(p[..., c], smooth) for c in range(p.shape[-1])], -1)
    return p.argmax(-1)


# Part 1 ------------------------------------------------------------------------

def run_method(name, param, image, models=None):
    """Return an (H, W) int segment map."""
    from scipy import ndimage
    from skimage import graph, segmentation
    from skimage.color import rgb2gray, rgb2lab
    from skimage.filters import sobel

    if name == "slic_lab":
        return segmentation.slic(image, n_segments=param, compactness=10, start_label=0)
    if name == "slic_hed":
        return segmentation.slic(hed_features(image), n_segments=param, compactness=0.1,
                                 channel_axis=-1, convert2lab=False, start_label=0)
    if name == "felzenszwalb":
        return segmentation.felzenszwalb(image, scale=param, sigma=0.8, min_size=50)
    if name == "watershed":
        gradient = sobel(rgb2gray(image))
        return segmentation.watershed(gradient, markers=param, compactness=1e-4)
    if name == "quickshift":
        return segmentation.quickshift(image, kernel_size=param, max_dist=param * 2, ratio=0.5)
    if name == "meanshift":
        # Mean shift is O(n^2)-ish; cluster a 64x64 thumbnail in (Lab, x, y).
        from skimage.transform import resize
        from sklearn.cluster import MeanShift

        small = resize(image, (64, 64), anti_aliasing=True)
        lab = rgb2lab(small) / np.array([10.0, 10.0, 10.0])
        yy, xx = np.mgrid[0:64, 0:64] / 8.0
        x = np.concatenate([lab.reshape(-1, 3), np.c_[yy.ravel(), xx.ravel()]], 1)
        labels = MeanShift(bandwidth=param, bin_seeding=True).fit_predict(x).reshape(64, 64)
        labels = np.kron(labels, np.ones((image.shape[0] // 64,) * 2, int))
        return _components(labels)
    if name in ("kmeans_cc", "gmm_cc"):
        labels = cluster_map(models[(name.split("_")[0], param)], image)
        return _components(labels)
    if name == "rag_merge":
        base = segmentation.slic(image, n_segments=1000, compactness=10, start_label=0)
        rag = graph.rag_mean_color(image, base)
        return graph.merge_hierarchical(base, rag, thresh=param, rag_copy=False,
                                        in_place_merge=True, merge_func=_merge_mean,
                                        weight_func=_weight_mean)
    if name == "ncut":
        base = segmentation.slic(image, n_segments=param, compactness=10, start_label=0)
        rag = graph.rag_mean_color(image, base, mode="similarity")
        return graph.cut_normalized(base, rag)
    raise ValueError(name)


def _components(labels):
    """Split a cluster-id map into connected segments."""
    from skimage.measure import label

    return label(labels + 1, connectivity=1)


def _weight_mean(graph, src, dst, n):
    diff = graph.nodes[dst]["mean color"] - graph.nodes[n]["mean color"]
    return {"weight": np.linalg.norm(diff)}


def _merge_mean(graph, src, dst):
    g = graph.nodes[dst]
    g["total color"] += graph.nodes[src]["total color"]
    g["pixel count"] += graph.nodes[src]["pixel count"]
    g["mean color"] = g["total color"] / g["pixel count"]


METHODS = [
    ("slic_lab", [100, 300, 1000]),
    ("slic_hed", [100, 300, 1000]),
    ("felzenszwalb", [1500, 400, 100]),
    ("watershed", [100, 300, 1000]),
    ("quickshift", [8, 5, 3]),
    ("meanshift", [1.2, 0.9, 0.6]),
    ("kmeans_cc", [5, 8, 16]),
    ("gmm_cc", [5, 8, 16]),
    ("rag_merge", [30, 18, 10]),
    ("ncut", [200, 400]),
]

_MODELS = None


def _init(models):
    global _MODELS
    _MODELS = models


def _score_tile(job):
    index, image, mask = job
    rows = []
    for name, params in METHODS:
        for param in params:
            start = time.time()
            try:
                seg = run_method(name, param, image, _MODELS)
            except Exception as e:  # e.g. ARPACK not converging inside cut_normalized
                rows.append((name, param, None, repr(e)))
                continue
            seconds = time.time() - start
            s = segment_scores(seg, mask)
            rows.append((name, param, s, seconds))
    with open(os.path.join(TILE_CACHE, f"{index}.pkl"), "wb") as f:
        pickle.dump(rows, f)
    return rows


def part1(n_tiles, workers):
    import pandas as pd

    _, masks256, names = load_split("val", 256)
    picks = sample_tiles(masks256, names, n_tiles)
    root = os.environ.get("BCSS_ROOT", os.path.join(REPO, "data", "bcss"))
    data = BCSSDataset(root=root, variant="512", split="val", lut=LUT)
    # Per-tile results are cached so an interrupted run resumes; delete
    # data/outputs/classical_tiles/ after changing METHODS.
    os.makedirs(TILE_CACHE, exist_ok=True)
    cached = lambda i: os.path.join(TILE_CACHE, f"{int(i)}.pkl")
    jobs = []
    for i in picks:
        if os.path.exists(cached(i)):
            continue
        s = data[int(i)]
        assert s.info.name == names[i]
        jobs.append((int(i), macenko_normalize(s.image), s.mask))
    print(f"part 1: {len(picks)} val tiles ({len(picks) - len(jobs)} cached), "
          f"{sum(len(p) for _, p in METHODS)} configs", flush=True)

    models = fit_pixel_models(normalize=True)
    if jobs:
        with Pool(workers, initializer=_init, initargs=(models,)) as pool:
            for t, _ in enumerate(pool.imap_unordered(_score_tile, jobs)):
                print(f"  tile {t + 1}/{len(jobs)}", flush=True)

    agg = {}
    for i in picks:
        with open(cached(i), "rb") as f:
            rows = pickle.load(f)
        for name, param, s, seconds in rows:
            a = agg.setdefault((name, param), {"n": [], "asa": [], "br": [], "sec": [],
                                               "failed": 0,
                                               "cm": np.zeros((NUM_CLASSES,) * 2, np.int64)})
            if s is None:
                a["failed"] += 1
                print(f"  {name} {param} failed: {seconds}", flush=True)
                continue
            a["n"].append(s["n_segments"])
            a["asa"].append(s["asa"])
            a["br"].append(s["boundary_recall"])
            a["sec"].append(seconds)
            a["cm"] += s["oracle_cm"]

    rows = []
    for (name, param), a in agg.items():
        rows.append({"method": name, "param": param,
                     "segments_per_tile": np.mean(a["n"]), "ASA": np.mean(a["asa"]),
                     "boundary_recall": np.nanmean(a["br"]),
                     "oracle_mIoU": class_report(a["cm"])["mIoU"],
                     "sec_per_tile": np.mean(a["sec"]), "failed_tiles": a["failed"]})
    df = pd.DataFrame(rows).sort_values(["method", "segments_per_tile"])
    df.to_csv(os.path.join(RESULTS_DIR, "clustering_superpixels.csv"), index=False,
              float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plot_asa(df)
    examples = []
    for i in picks[:3]:
        s = data[int(i)]
        examples.append((macenko_normalize(s.image), s.mask))
    plot_examples(examples, models)
    return df


def plot_asa(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in df.groupby("method"):
        g = g.sort_values("segments_per_tile")
        axes[0].plot(g.segments_per_tile, g.ASA, "o-", label=name)
        axes[1].plot(g.segments_per_tile, g.boundary_recall, "o-", label=name)
    for ax, title in zip(axes, ["ASA (achievable accuracy)", "Boundary recall (2 px)"]):
        ax.set_xscale("log")
        ax.set_xlabel("segments per 512x512 tile")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "asa_vs_segments.png"), dpi=130)


def plot_examples(jobs, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    shown = [("slic_lab", 300), ("slic_hed", 300), ("felzenszwalb", 400), ("watershed", 300),
             ("quickshift", 5), ("meanshift", 0.9), ("kmeans_cc", 8), ("gmm_cc", 8),
             ("rag_merge", 18), ("ncut", 400)]
    colors = np.array([[220, 50, 50], [240, 160, 200], [40, 90, 220], [30, 30, 30],
                       [60, 180, 80]], np.uint8)
    fig, axes = plt.subplots(len(jobs), len(shown) + 2, figsize=(2.2 * (len(shown) + 2),
                                                                 2.3 * len(jobs)))
    for r, (image, mask) in enumerate(jobs):
        gt = np.where(mask[..., None] == IGNORE_INDEX, 255, colors[np.minimum(mask, 4)])
        axes[r, 0].imshow(image)
        axes[r, 1].imshow(gt.astype(np.uint8))
        for c, (name, param) in enumerate(shown):
            try:
                seg = run_method(name, param, image, models)
                axes[r, c + 2].imshow(mark_boundaries(image, seg, color=(1, 1, 0)))
            except Exception:
                axes[r, c + 2].imshow(image)
                axes[r, c + 2].text(5, 30, "failed", color="red")
            if r == 0:
                axes[r, c + 2].set_title(f"{name}\n{param}", fontsize=8)
    axes[0, 0].set_title("image (Macenko)", fontsize=8)
    axes[0, 1].set_title("ground truth\nred tumor, pink stroma,\nblue infl., black necr.",
                         fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "clustering_examples.png"), dpi=110)


# Part 2 ------------------------------------------------------------------------

def _label_maps(args):
    model, images = args
    return np.stack([cluster_map(model, im) for im in images])


def part2(workers):
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    tr_x, tr_y, tr_names = load_split("train", 256)
    fit = np.flatnonzero(~dev_split(tr_names))
    name_tiles = np.random.default_rng(1).choice(fit, 800, replace=False)
    va_x, va_y, _ = load_split("val", 256)

    tr_x, tr_y = tr_x[name_tiles], tr_y[name_tiles]

    rows = []
    for normalize in (True, False):
        # One result file per setting so an interrupted run resumes.
        cache = os.path.join(OUTPUT_DIR, f"classical_semantic_macenko{int(normalize)}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                rows += pickle.load(f)
            continue
        setting_rows = []
        tr_in = np.stack([macenko_normalize(im) for im in tr_x]) if normalize else tr_x
        va_in = np.stack([macenko_normalize(im) for im in va_x]) if normalize else va_x
        models = fit_pixel_models(normalize=normalize)
        for (kind, k), model in models.items():
            start = time.time()
            with Pool(workers) as pool:
                tr_maps = np.concatenate(pool.map(
                    _label_maps, [(model, c) for c in np.array_split(tr_in, workers * 4)]))
                va_maps = np.concatenate(pool.map(
                    _label_maps, [(model, c) for c in np.array_split(va_in, workers * 4)]))
            # Cluster x class counts on train tiles -> names for clusters.
            valid = tr_y != IGNORE_INDEX
            table = np.zeros((k, NUM_CLASSES), np.int64)
            np.add.at(table, (tr_maps[valid], tr_y[valid]), 1)
            majority = table.argmax(1)
            cm = confusion_matrix(majority[va_maps], va_y, NUM_CLASSES)
            row = {"method": kind, "k": k, "macenko": normalize, "mapping": "majority",
                   **class_report(cm), "sec_total": time.time() - start}
            setting_rows.append(row)
            if k == NUM_CLASSES:
                r, c = linear_sum_assignment(-table)
                hung = np.zeros(k, int)
                hung[r] = c
                cm = confusion_matrix(hung[va_maps], va_y, NUM_CLASSES)
                setting_rows.append({**row, "mapping": "hungarian", **class_report(cm)})
            print(f"  {kind} k={k} macenko={normalize}: mIoU {row['mIoU']:.3f}", flush=True)
        with open(cache, "wb") as f:
            pickle.dump(setting_rows, f)
        rows += setting_rows
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "clustering_semantic.csv"), index=False,
              float_format="%.4f")
    cols = ["method", "k", "macenko", "mapping", "pixel_acc", "mIoU"] + \
        [f"IoU_{n}" for n in CLASS_NAMES]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--part", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.part in ("1", "all"):
        part1(args.tiles, args.workers)
    if args.part in ("2", "all"):
        part2(args.workers)


if __name__ == "__main__":
    main()
