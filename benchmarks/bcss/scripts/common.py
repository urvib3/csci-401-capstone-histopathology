# Shared helpers for the BCSS clustering and baseline experiments.
#
# - Cached, downsampled arrays of the BCSS_512 splits with coarse labels.
# - A dev split: train ROIs held out for model selection, so the official
#   val split (held-out hospitals) is only used for final numbers.
# - Macenko stain normalization and HED features.
# - Segmentation quality metrics for over-segmentations.

import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from bcss import IGNORE_INDEX, BCSSDataset, coarse_lut  # noqa: E402
from bcss.metrics import confusion_matrix  # noqa: E402

CACHE_DIR = os.path.join(REPO, "data", "cache")
OUTPUT_DIR = os.path.join(REPO, "data", "outputs")
RESULTS_DIR = os.path.join(REPO, "benchmarks", "bcss", "results")

LUT, CLASS_NAMES = coarse_lut()  # tumor, stroma, inflammatory, necrosis, other
NUM_CLASSES = len(CLASS_NAMES)


def load_split(split, size=256):
    """
    BCSS_512 split with coarse labels, resized to size x size.

    Images use bilinear resampling, masks nearest. Cached under data/cache
    on first use (~0.4 GB per 1000 tiles at 512, ~0.25 GB at 256).

    returns: images (N, size, size, 3) uint8, masks (N, size, size) uint8,
             names (N,) str
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"bcss512_{split}_{size}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["images"], z["masks"], z["names"]

    root = os.environ.get("BCSS_ROOT", os.path.join(REPO, "data", "bcss"))
    data = BCSSDataset(root=root, variant="512", split=split, lut=LUT)
    n = len(data)
    images = np.empty((n, size, size, 3), np.uint8)
    masks = np.empty((n, size, size), np.uint8)
    start = time.time()
    for i in range(n):
        s = data[i]
        image, mask = s.image, s.mask
        if size != image.shape[0]:
            image = np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))
            mask = np.asarray(Image.fromarray(mask).resize((size, size), Image.NEAREST))
        images[i], masks[i] = image, mask
        if i % 1000 == 0:
            print(f"  caching {split}@{size}: {i}/{n} ({time.time() - start:.0f}s)", flush=True)
    names = np.array([t.name for t in data.tiles])
    np.savez(path, images=images, masks=masks, names=names)
    return images, masks, names


def roi_of(name):
    """ROI id (<slide>_xmin<x>_ymin<y>) from a tile file name."""
    return name.split("_MPP")[0]


def dev_split(names, every=8):
    """
    Boolean mask over train tiles: True for tiles in held-out dev ROIs.

    Every `every`-th ROI (sorted by id) is held out, so dev tiles never share
    an ROI with the tiles used for fitting.
    """
    rois = sorted({roi_of(n) for n in names})
    held = set(rois[::every])
    return np.array([roi_of(n) in held for n in names])


def sample_tiles(masks, names, n, seed=0, min_labeled=0.5):
    """
    Pick n tile indices spread across ROIs (round robin over shuffled ROIs),
    skipping tiles with less than min_labeled of their pixels labeled.
    """
    rng = np.random.default_rng(seed)
    labeled = (masks != IGNORE_INDEX).mean(axis=(1, 2))
    by_roi = {}
    for i, name in enumerate(names):
        if labeled[i] >= min_labeled:
            by_roi.setdefault(roi_of(name), []).append(i)
    pools = [rng.permutation(v).tolist() for _, v in sorted(by_roi.items())]
    rng.shuffle(pools)
    picked = []
    while len(picked) < n and any(pools):
        for pool in pools:
            if pool and len(picked) < n:
                picked.append(pool.pop())
    return np.array(sorted(picked))


# Stain handling --------------------------------------------------------------

# Reference H&E stain vectors and max concentrations commonly used with
# Macenko et al. 2009 (values from the reference implementation by M. Macenko /
# schaugf/HEnorm_python).
HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAX_C_REF = np.array([1.9705, 1.0308])


def macenko_normalize(image, io=240, alpha=1, beta=0.15):
    """
    Map an RGB uint8 H&E image onto the reference stain basis.

    Falls back to the input when the tile has too little tissue to estimate
    stain vectors.
    """
    h, w, _ = image.shape
    rgb = image.reshape(-1, 3).astype(np.float64)
    od = -np.log((rgb + 1) / io)
    tissue = od[~np.any(od < beta, axis=1)]
    if len(tissue) < 100:
        return image
    _, eigvecs = np.linalg.eigh(np.cov(tissue.T))
    plane = tissue @ eigvecs[:, 1:3]
    phi = np.arctan2(plane[:, 1], plane[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v_min = eigvecs[:, 1:3] @ np.array([np.cos(lo), np.sin(lo)])
    v_max = eigvecs[:, 1:3] @ np.array([np.cos(hi), np.sin(hi)])
    # Hematoxylin is the vector with the larger first (red OD) component.
    he = np.array([v_min, v_max]).T if v_min[0] > v_max[0] else np.array([v_max, v_min]).T
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    max_c = np.percentile(conc, 99, axis=1)
    conc *= (MAX_C_REF / np.maximum(max_c, 1e-6))[:, None]
    out = io * np.exp(-HE_REF @ conc)
    return np.clip(out.T, 0, 255).reshape(h, w, 3).astype(np.uint8)


def hed_features(image):
    """(H, W, 3) float32 hematoxylin / eosin / DAB optical densities, rescaled to ~[0, 1]."""
    from skimage.color import rgb2hed

    hed = rgb2hed(image).astype(np.float32)
    lo = np.array([-0.70, -0.10, -0.55], np.float32)  # rough ranges for H&E tiles
    hi = np.array([-0.15, 0.40, -0.20], np.float32)
    return np.clip((hed - lo) / (hi - lo), 0, 1)


# Over-segmentation metrics -----------------------------------------------------

def segment_scores(segments, mask, tolerance=2):
    """
    Quality of one over-segmentation against a coarse ground-truth mask.

    segments: (H, W) int segment ids. mask: (H, W) class ids / IGNORE_INDEX.

    returns dict:
        n_segments
        asa:        achievable segmentation accuracy, the pixel accuracy of
                    giving each segment its majority ground-truth class (an
                    upper bound for any labeling of these segments, e.g. by
                    a WCSP)
        boundary_recall: fraction of ground-truth class-boundary pixels within
                    `tolerance` px of a segment boundary (NaN if none)
        oracle_cm:  confusion matrix of the majority labeling, summed over
                    tiles to get an oracle mIoU
    """
    from scipy import ndimage

    seg = np.unique(segments, return_inverse=True)[1].reshape(segments.shape)
    n_seg = int(seg.max()) + 1
    valid = mask != IGNORE_INDEX
    counts = np.zeros((n_seg, NUM_CLASSES), np.int64)
    np.add.at(counts, (seg[valid], mask[valid]), 1)
    majority = counts.argmax(axis=1)
    oracle = majority[seg]
    total = valid.sum()
    asa = counts.max(axis=1).sum() / total if total else np.nan

    def edges(labels):
        e = np.zeros(labels.shape, bool)
        e[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        e[1:, :] |= labels[1:, :] != labels[:-1, :]
        return e

    gt = np.where(valid, mask, 255).astype(np.int32)
    gt_edges = edges(gt)
    # Only edges between two labeled classes; ROI borders are not tissue boundaries.
    inner = ndimage.binary_erosion(valid, iterations=1)
    gt_edges &= inner
    if gt_edges.sum():
        near = ndimage.binary_dilation(edges(seg), iterations=tolerance)
        recall = (gt_edges & near).sum() / gt_edges.sum()
    else:
        recall = np.nan
    return {
        "n_segments": n_seg,
        "asa": float(asa),
        "boundary_recall": float(recall),
        "oracle_cm": confusion_matrix(oracle, mask, NUM_CLASSES),
    }


def class_report(cm):
    """Per-class IoU / Dice plus pixel accuracy and mIoU from a confusion matrix."""
    from bcss.metrics import dice_per_class, iou_per_class, mean_iou, pixel_accuracy

    iou, dice = iou_per_class(cm), dice_per_class(cm)
    row = {"pixel_acc": pixel_accuracy(cm), "mIoU": mean_iou(cm),
           "mDice": float(np.nanmean(dice))}
    for name, i, d in zip(CLASS_NAMES, iou, dice):
        row[f"IoU_{name}"] = float(i)
        row[f"Dice_{name}"] = float(d)
    return row
