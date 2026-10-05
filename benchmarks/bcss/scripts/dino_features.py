# DINO-family patch features for BCSS_512: unsupervised clustering and a
# linear-probe baseline.
#
#   .venv/bin/python benchmarks/bcss/scripts/dino_features.py extract --model dinov2
#   .venv/bin/python benchmarks/bcss/scripts/dino_features.py extract --model phikon2
#   .venv/bin/python benchmarks/bcss/scripts/dino_features.py evaluate
#
# Encoders (frozen, via Hugging Face transformers):
#   dinov2  facebook/dinov2-base  ViT-B/14, DINOv2 on natural images (LVD-142M)
#   phikon2 owkin/phikon-v2       ViT-L/16, DINOv2 recipe on ~460M pathology tiles
#
# Each 512x512 tile is resized to the encoder's input (224 for the /14 DINOv2,
# 256 for the /16 Phikon-v2) so both give a 16x16 grid of patch tokens, i.e.
# one feature vector per 32x32 px cell of the original tile. Labels per cell = majority ground-truth class.
#
# evaluate runs, per encoder:
#   1. Unsupervised segmentation: per-tile k-means and Leiden on the token
#      graph -> segment scores (ASA, oracle mIoU, boundary recall) on the
#      same 60 val tiles as cluster_classical.py.
#   2. Unsupervised classification: a global k-means codebook fit on train
#      tokens (no labels); clusters named by majority train label; val mIoU.
#   3. Linear probe: logistic regression on frozen tokens (train fit tiles,
#      C picked on dev ROIs); val mIoU. Scored on the 16x16 cell grid and on
#      pixels (cell predictions upsampled to 256x256 masks).

import argparse
import json
import os
import time

import numpy as np

from common import (CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, OUTPUT_DIR, RESULTS_DIR,
                    class_report, confusion_matrix, dev_split, load_split,
                    sample_tiles, segment_scores)

ENCODERS = {
    "dinov2": {"hf": "facebook/dinov2-base", "input": 224, "patch": 14},
    "phikon2": {"hf": "owkin/phikon-v2", "input": 256, "patch": 16},
}
GRID = 16  # tokens per side for both encoders
FEAT_DIR = os.path.join(OUTPUT_DIR, "dino")
N_FIT_TILES = 2000  # train tiles (outside dev ROIs) used for codebook / probe
CHUNK = 400  # tiles per resumable extraction chunk


def tile_sets(seed=0):
    """Indices of the train-fit subset, dev tiles and all val tiles."""
    _, tr_y, tr_names = load_split("train", 256)
    dev = dev_split(tr_names)
    rng = np.random.default_rng(seed)
    fit = np.sort(rng.choice(np.flatnonzero(~dev), N_FIT_TILES, replace=False))
    return {"fit": fit, "dev": np.flatnonzero(dev)}


def cell_labels(masks, grid=GRID):
    """Majority class per cell; IGNORE_INDEX if under half the cell is labeled."""
    n, h, w = masks.shape
    c = h // grid
    blocks = masks.reshape(n, grid, c, grid, c).transpose(0, 1, 3, 2, 4).reshape(n, grid, grid, -1)
    counts = np.stack([(blocks == k).sum(-1) for k in range(NUM_CLASSES)], -1)
    labels = counts.argmax(-1).astype(np.uint8)
    labels[counts.sum(-1) < c * c / 2] = IGNORE_INDEX
    return labels


# Extraction ----------------------------------------------------------------------

def extract(model_key, batch=16):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel

    spec = ENCODERS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(spec["hf"]).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    os.makedirs(FEAT_DIR, exist_ok=True)
    sets = tile_sets()
    jobs = [("train", "fit", sets["fit"]), ("train", "dev", sets["dev"]), ("val", "val", None)]
    for split, tag, idx in jobs:
        path = os.path.join(FEAT_DIR, f"{model_key}_{tag}.npz")
        if os.path.exists(path):
            print("exists", path)
            continue
        images, masks, names = load_split(split, 256)
        if idx is not None:
            images, masks, names = images[idx], masks[idx], names[idx]
        # Written in chunks of CHUNK tiles so an interrupted run resumes where it stopped.
        start = time.time()
        parts = []
        for c in range(0, len(images), CHUNK):
            part = os.path.join(FEAT_DIR, f"{model_key}_{tag}.part{c // CHUNK:03d}.npy")
            parts.append(part)
            if os.path.exists(part):
                continue
            chunk = images[c:c + CHUNK]
            feats = np.empty((len(chunk), GRID, GRID, model.config.hidden_size), np.float16)
            with torch.no_grad():
                for i in range(0, len(chunk), batch):
                    x = torch.from_numpy(chunk[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255
                    x = F.interpolate(x, size=spec["input"], mode="bilinear", align_corners=False)
                    out = model(pixel_values=(x - mean) / std).last_hidden_state
                    tokens = out[:, -GRID * GRID:]  # drop CLS (and any register tokens)
                    feats[i:i + batch] = tokens.reshape(-1, GRID, GRID, tokens.shape[-1]).cpu().numpy()
            np.save(part, feats)
            print(f"  {model_key} {tag}: {c + len(chunk)}/{len(images)} "
                  f"({time.time() - start:.0f}s)", flush=True)
        feats = np.concatenate([np.load(p) for p in parts])
        np.savez(path, feats=feats, cells=cell_labels(masks), names=names)
        for part in parts:
            os.remove(part)
        print(f"saved {path} ({time.time() - start:.0f}s)", flush=True)


def load_feats(model_key, tag):
    z = np.load(os.path.join(FEAT_DIR, f"{model_key}_{tag}.npz"))
    return z["feats"].astype(np.float32), z["cells"], z["names"]


# Evaluation ----------------------------------------------------------------------

def l2norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-6)


def upsample_cells(cell_pred, size=256):
    r = size // cell_pred.shape[-1]
    return np.repeat(np.repeat(cell_pred, r, axis=-2), r, axis=-1)


def leiden_segments(tokens, resolution, k=8):
    """Leiden communities of a kNN graph over one tile's tokens (cosine)."""
    import igraph as ig
    import leidenalg

    x = l2norm(tokens.reshape(-1, tokens.shape[-1]))
    sim = x @ x.T
    np.fill_diagonal(sim, -1)
    nbrs = np.argsort(-sim, axis=1)[:, :k]
    edges = {(min(i, j), max(i, j)) for i in range(len(x)) for j in nbrs[i]}
    edges = list(edges)
    weights = [max(float(sim[i, j]), 0.0) for i, j in edges]
    g = ig.Graph(n=len(x), edges=edges)
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    weights=weights, resolution_parameter=resolution, seed=0)
    return np.array(part.membership).reshape(GRID, GRID)


def unsupervised_segmentation(model_key, val_feats, picks, va_masks):
    """Per-tile k-means / Leiden on tokens, scored like the classical methods."""
    from skimage.measure import label
    from sklearn.cluster import KMeans

    rows = []
    configs = [("kmeans", k) for k in (4, 8, 16)] + [("leiden", r) for r in (0.5, 1.0, 2.0)]
    for method, param in configs:
        agg = {"n": [], "asa": [], "br": [], "cm": np.zeros((NUM_CLASSES,) * 2, np.int64)}
        start = time.time()
        for i in picks:
            tokens = val_feats[i]
            if method == "kmeans":
                x = l2norm(tokens.reshape(-1, tokens.shape[-1]))
                clusters = KMeans(param, n_init=4, random_state=0).fit_predict(x).reshape(GRID, GRID)
            else:
                clusters = leiden_segments(tokens, param)
            seg = label(upsample_cells(clusters, 256) + 1, connectivity=1)
            s = segment_scores(seg, va_masks[i])
            agg["n"].append(s["n_segments"])
            agg["asa"].append(s["asa"])
            agg["br"].append(s["boundary_recall"])
            agg["cm"] += s["oracle_cm"]
        rows.append({"method": f"{model_key}_{method}_per_tile", "param": param,
                     "segments_per_tile": np.mean(agg["n"]), "ASA": np.mean(agg["asa"]),
                     "boundary_recall": np.nanmean(agg["br"]),
                     "oracle_mIoU": class_report(agg["cm"])["mIoU"],
                     "sec_per_tile": (time.time() - start) / len(picks)})
    return rows


def score(cell_pred, val_cells, va_masks):
    """Scores on the token grid and on 256x256 pixels."""
    cm_cells = confusion_matrix(cell_pred, val_cells, NUM_CLASSES)
    cm_pix = np.zeros((NUM_CLASSES,) * 2, np.int64)
    for p, m in zip(cell_pred, va_masks):
        cm_pix += confusion_matrix(upsample_cells(p), m, NUM_CLASSES)
    return class_report(cm_cells), class_report(cm_pix), cm_pix


def codebook(model_key, fit, val, va_masks):
    from sklearn.cluster import MiniBatchKMeans

    fx, fy, _ = fit
    vx, vy, _ = val
    d = fx.shape[-1]
    x = l2norm(fx.reshape(-1, d))
    y = fy.ravel()
    rows = []
    for k in (5, 10, 20, 50):
        km = MiniBatchKMeans(k, random_state=0, n_init=3, batch_size=8192).fit(x)
        labeled = y != IGNORE_INDEX
        table = np.zeros((k, NUM_CLASSES), np.int64)
        np.add.at(table, (km.labels_[labeled], y[labeled]), 1)
        names = table.argmax(1)
        pred = names[km.predict(l2norm(vx.reshape(-1, d)))].reshape(vy.shape)
        rc, rp, _ = score(pred, vy, va_masks)
        rows.append({"encoder": model_key, "method": "kmeans_codebook", "k": k,
                     "cells_mIoU": rc["mIoU"], **rp})
        print(f"  {model_key} codebook k={k}: pixel mIoU {rp['mIoU']:.3f}", flush=True)
    return rows


def linear_probe(model_key, fit, dev, val, va_masks):
    import torch

    def tensors(split):
        x, y, _ = split
        x = torch.from_numpy(l2norm(x.reshape(-1, x.shape[-1])))
        y = torch.from_numpy(y.ravel().astype(np.int64))
        keep = y != IGNORE_INDEX
        return x, y, keep

    fx, fy, fk = tensors(fit)
    dx, dy, dk = tensors(dev)
    vx, _, _ = tensors(val)
    counts = torch.bincount(fy[fk], minlength=NUM_CLASSES).float()
    weights = (1 / counts.sqrt())
    weights = weights / weights.mean()

    def train(wd, epochs=30):
        torch.manual_seed(0)
        layer = torch.nn.Linear(fx.shape[1], NUM_CLASSES)
        opt = torch.optim.AdamW(layer.parameters(), lr=1e-2, weight_decay=wd)
        x, y = fx[fk], fy[fk]
        for _ in range(epochs):
            perm = torch.randperm(len(x))
            for s in range(0, len(x), 8192):
                b = perm[s:s + 8192]
                loss = torch.nn.functional.cross_entropy(layer(x[b] * 10), y[b], weight=weights)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return layer

    best = None
    for wd in (1e-4, 1e-2, 1e-1):
        layer = train(wd)
        with torch.no_grad():
            pred = layer(dx * 10).argmax(1).numpy()
        cm = confusion_matrix(pred[dk.numpy()], dy[dk].numpy(), NUM_CLASSES)
        dev_miou = class_report(cm)["mIoU"]
        print(f"  {model_key} probe wd={wd}: dev cell mIoU {dev_miou:.3f}", flush=True)
        if best is None or dev_miou > best[0]:
            best = (dev_miou, wd, layer)
    dev_miou, wd, layer = best
    with torch.no_grad():
        logits = layer(vx * 10)
    vy = val[1]
    pred = logits.argmax(1).numpy().reshape(vy.shape)
    probs = logits.softmax(1).numpy().reshape(*vy.shape, NUM_CLASSES)
    np.savez_compressed(os.path.join(FEAT_DIR, f"{model_key}_probe_val_probs16.npz"),
                        probs=probs.astype(np.float16), names=val[2])
    rc, rp, cm = score(pred, vy, va_masks)
    return {"encoder": model_key, "method": "linear_probe", "k": None, "weight_decay": wd,
            "dev_cells_mIoU": dev_miou, "cells_mIoU": rc["mIoU"], **rp}, cm


def evaluate(models):
    import pandas as pd

    _, va_masks, va_names = load_split("val", 256)
    picks = sample_tiles(va_masks, va_names, 60)
    seg_rows, cls_rows, cms = [], [], {}
    for key in models:
        fit, dev, val = (load_feats(key, t) for t in ("fit", "dev", "val"))
        assert (val[2] == va_names).all()
        print(f"{key}: fit {fit[0].shape}, dev {dev[0].shape}, val {val[0].shape}", flush=True)
        seg_rows += unsupervised_segmentation(key, val[0], picks, va_masks)
        cls_rows += codebook(key, fit, val, va_masks)
        row, cm = linear_probe(key, fit, dev, val, va_masks)
        cls_rows.append(row)
        cms[key] = cm.tolist()
        print(f"  {key} linear probe: val pixel mIoU {row['mIoU']:.3f}", flush=True)
        del fit, dev, val  # free ~5 GB before loading the next encoder

    seg = pd.DataFrame(seg_rows)
    seg.to_csv(os.path.join(RESULTS_DIR, "dino_segmentation.csv"), index=False, float_format="%.4f")
    cls = pd.DataFrame(cls_rows)
    cls.to_csv(os.path.join(RESULTS_DIR, "dino_classification.csv"), index=False,
               float_format="%.4f")
    with open(os.path.join(RESULTS_DIR, "dino_probe_confusion.json"), "w") as f:
        json.dump({"rows=truth, cols=pred": CLASS_NAMES, **cms}, f, indent=1)
    print(seg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cols = ["encoder", "method", "k", "cells_mIoU", "pixel_acc", "mIoU"] + \
        [f"IoU_{n}" for n in CLASS_NAMES]
    print(cls[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract", "evaluate"])
    parser.add_argument("--model", choices=list(ENCODERS), action="append")
    args = parser.parse_args()
    models = args.model or list(ENCODERS)
    if args.command == "extract":
        for m in models:
            extract(m)
    else:
        evaluate(models)


if __name__ == "__main__":
    main()
