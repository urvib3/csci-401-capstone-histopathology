"""WCSP using only raw type probabilities and their foreground components."""
import argparse
import sys

import numpy as np
from scipy.ndimage import label

from prepare_fresh import CACHE, HERE, PREVIOUS, source

sys.path.insert(0, str(PREVIOUS))
from run import confusion, iou, score


def inputs(rows):
    for row in rows:
        i = row["image_index"]
        with np.load(CACHE / f"pred_{i:04d}.npz") as data:
            p = data["p"]
        with np.load(CACHE / f"gt_{i:04d}.npz") as data:
            gt = data["types"][data["instances"]]
        raw = p.argmax(-1)
        connected, n = label(raw > 0)
        foreground = raw > 0
        ids = connected[foreground] - 1
        probabilities = p[foreground, 1:].astype(float)
        best = raw[foreground] - 1
        costs = -np.log(np.maximum(probabilities, 1e-12))
        costs -= costs[np.arange(len(best)), best, None]
        ranks = np.partition(probabilities, -2, axis=1)
        margin = ranks[:, -1] - ranks[:, -2]
        sizes = np.bincount(ids, minlength=n)
        yield gt, raw, foreground, ids, costs, margin, sizes


def refine(patch, strength, limit, adaptive):
    _, raw, foreground, ids, costs, margin, sizes = patch
    if not len(ids) or strength == 0:
        return raw.copy()
    gate = sizes[ids] <= limit
    weight = strength * gate * ((1 - margin)**2 if adaptive else 1)
    # If the component label is k, each pixel independently chooses between
    # taking k and keeping its lowest-unary type while paying the mismatch.
    energy = np.zeros((len(sizes), 5))
    np.add.at(energy, ids, np.minimum(costs, weight[:, None]))
    center = energy.argmin(1)[ids]
    switch = costs[np.arange(len(ids)), center] < weight - 1e-12
    output = raw.copy()
    output[foreground] = np.where(switch, center + 1, raw[foreground])
    assert np.array_equal(output > 0, foreground)
    return output


def evaluate(patches, cfg):
    matrices = []
    fixes = breaks = 0
    raw_matrices = []
    for patch in patches:
        gt, raw = patch[:2]
        output = refine(patch, **cfg)
        matrices.append(confusion(gt, output))
        raw_matrices.append(confusion(gt, raw))
        fixes += int((~(raw == gt) & (output == gt)).sum())
        breaks += int(((raw == gt) & ~(output == gt)).sum())
    b, a = np.stack(matrices), np.stack(raw_matrices)
    result = dict(config=cfg, bio_miou=float(score(b.sum(0))), fixes=fixes, breaks=breaks)
    return result, a, b


def main(stage):
    source.HERE = HERE
    split = source.load(HERE / "split.json")
    patches = list(inputs(split["dev" if stage == "dev" else "test"]))
    control = dict(strength=0, limit=256, adaptive=False)
    if stage == "dev":
        options = [control] + [dict(strength=strength, limit=limit, adaptive=adaptive)
                              for strength in (.10, .25, .50, 1., 2.)
                              for limit in (128, 256, 65536) for adaptive in (False, True)]
        rows = [evaluate(patches, item)[0] for item in options]
        best = max(rows, key=lambda row: row["bio_miou"])
        source.write("raw_only_dev.json", rows)
        source.write("raw_only_frozen.json", best["config"])
        print("Raw-only dev baseline:", rows[0], "selected:", best, flush=True)
        return
    cfg = source.load(HERE / "raw_only_frozen.json")
    result, raw, out = evaluate(patches, cfg)
    draws = np.random.default_rng(20261116).integers(len(patches), size=(5000, len(patches)))
    delta = score(out[draws].sum(1)) - score(raw[draws].sum(1))
    patch_diff = score(out) - score(raw)
    result.update(raw_miou=float(score(raw.sum(0))), delta=float(score(out.sum(0)) - score(raw.sum(0))),
                  ci95=np.percentile(delta, [2.5, 97.5]).tolist(),
                  per_class_delta=(iou(out.sum(0)) - iou(raw.sum(0)))[1:].tolist(),
                  wlt=[int((patch_diff > 1e-12).sum()), int((patch_diff < -1e-12).sum()),
                       int((abs(patch_diff) <= 1e-12).sum())])
    np.savez_compressed(HERE / "raw_only_confusions.npz", raw=raw, wcsp=out)
    source.write("raw_only_evaluation.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "final"])
    main(parser.parse_args().stage)
