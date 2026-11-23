# Check three candidate biological rules against the data before encoding them.
#
#   R1 continuity: tissue compartments are larger than a tile, so a tile's classes
#      match its neighboring tiles' (same ROI, offset 512 px).
#   R2 stromal immune cells: lymphocytic infiltrates sit in stroma, so an inflammatory
#      region's border is mostly stroma.
#   R3 confluent necrosis: tumor necrosis forms large compact areas, so small necrosis
#      fragments are rare.
#
# Rules are read from fit ground truth; val ground truth vs. the U-Net and the current
# WCSP (saved run) shows whether the outputs break them.
#
#   python experiments/bcss_wcsp/rules/check_rules.py

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import CONNECTIVE, DEAD, IGNORE, INFLAMMATORY, NUM_LABELS  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
SMALL = (1024, 4096)
OFFSETS = re.compile(r"(.*)_MPP-0_(\d+)_(\d+)_size512$")


def position(tile_id):
    roi, a, b = OFFSETS.match(tile_id).groups()
    return roi, int(a) // 512, int(b) // 512


def presence(labels, threshold=0.01, valid=None):
    ok = labels != IGNORE if valid is None else valid
    if ok.sum() < 1000:
        return None
    share = np.bincount(labels[ok], minlength=NUM_LABELS)[:NUM_LABELS] / ok.sum()
    return share >= threshold


def stroma_border_shares(labels, cls, min_area=64):
    """Per region of class cls: share of its outside border pixels (labeled) that are stroma."""
    comp, n = ndimage.label(labels == cls, structure=np.ones((3, 3)))
    out = []
    for r in range(1, n + 1):
        region = comp == r
        if region.sum() < min_area:
            continue
        ring = ndimage.binary_dilation(region) & ~region & (labels != IGNORE)
        if ring.sum() >= 10:
            out.append(float((labels[ring] == CONNECTIVE).mean()))
    return out


def necrosis_sizes(labels):
    comp, n = ndimage.label(labels == DEAD, structure=np.ones((3, 3)))
    return np.bincount(comp.ravel())[1:].tolist() if n else []


class Tally:
    def __init__(self):
        self.r1 = defaultdict(lambda: np.zeros((2, NUM_LABELS)))   # key -> [count, present]
        self.r2 = []
        self.r2_other = []
        self.r3 = []

    def add_tile(self, labels):
        self.r2 += stroma_border_shares(labels, INFLAMMATORY)
        self.r3 += necrosis_sizes(labels)

    def add_r1(self, mine, neighbors):
        """mine: (K,) bool; neighbors: list of (K,) bool."""
        if not neighbors:
            return
        any_n = np.any(neighbors, axis=0)
        for k in range(NUM_LABELS):
            key = "neighbor_has" if any_n[k] else "no_neighbor_has"
            self.r1[key][0, k] += 1
            self.r1[key][1, k] += mine[k]
            self.r1["all"][0, k] += 1
            self.r1["all"][1, k] += mine[k]

    def report(self):
        r1 = {key: {NAMES[k]: float(v[1, k] / max(v[0, k], 1)) for k in range(NUM_LABELS)}
              for key, v in self.r1.items()}
        r2 = np.array(self.r2)
        sizes = np.array(self.r3)
        return {
            "R1_P(class in tile)": r1,
            "R2_inflammatory_regions": len(r2),
            "R2_median_stroma_share_of_border": float(np.median(r2)) if len(r2) else None,
            "R2_share_regions_mostly_stroma(>=0.5)": float((r2 >= 0.5).mean()) if len(r2) else None,
            "R3_necrosis_regions": len(sizes),
            "R3_median_necrosis_region_px": float(np.median(sizes)) if len(sizes) else None,
            **{f"R3_share_necrosis_px_in_regions_<{a}": float(sizes[sizes < a].sum() / max(sizes.sum(), 1))
               for a in SMALL},
        }


def neighbors_of(pos, index):
    roi, a, b = pos
    return [index[(roi, a + da, b + db)] for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if (roi, a + da, b + db) in index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--run", default="valfull_data_lam8")
    parser.add_argument("--out", default="experiments/bcss_wcsp/rules/check_rules.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = {}

    # fit ground truth: where the rules come from
    fit = pipeline.dataset_for(cfg, "fit")
    present, tally = {}, Tally()
    for i in range(len(fit)):
        target = fit[i].grid_target
        tally.add_tile(target)
        p = presence(target)
        if p is not None:
            present[position(fit.ids[i])] = p
    for pos, p in present.items():
        tally.add_r1(p, neighbors_of(pos, present))
    out["fit_truth"] = tally.report()

    # val: ground truth vs. U-Net vs. current WCSP; neighbors from cached U-Net predictions
    val = pipeline.dataset_for(cfg, "val")
    store = pipeline.store_for(cfg, "val")
    unet_present = {}
    for tile_id in val.ids:
        if tile_id in store:
            p = presence(store.load(tile_id).probs.argmax(0), valid=np.ones((256, 256), bool))
            unet_present[position(tile_id)] = p
    tallies = {"truth": Tally(), "unet": Tally(), "wcsp": Tally()}
    for shard in sorted((cfg.runs_root / args.run / "val").glob("shard_*.npz")):
        z = np.load(shard)
        for tile_id, pred, refined in zip(z["ids"].tolist(), z["argmax"], z["wcsp"]):
            target = val[val.index(tile_id)].grid_target
            ignored = target == IGNORE
            truth_p = presence(target)
            if truth_p is None:
                continue
            nbrs = neighbors_of(position(tile_id), unet_present)
            for name, labels in (("truth", target), ("unet", np.where(ignored, IGNORE, pred)),
                                 ("wcsp", np.where(ignored, IGNORE, refined))):
                tallies[name].add_tile(labels)
                p = truth_p if name == "truth" else presence(labels)
                if p is not None:
                    tallies[name].add_r1(p, nbrs)
    for name, t in tallies.items():
        out[f"val_{name}"] = t.report()
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
