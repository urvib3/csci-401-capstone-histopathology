# How often each candidate rule holds in the ground truth, and how often the U-Net and
# the current WCSP break it. One number per rule per source, so a rule is only worth
# encoding when the truth follows it and the U-Net does not.
#
#   presence     tile's class set (classes with >= 1% of pixels) is rare (P_fit < 1%)
#   per_class    share of tiles containing each class (over-prediction -> label cost)
#   area         a class that is present covers < 2% of the tile
#   fragments    pieces per class per tile; share of class pixels in pieces < 256 px
#   infl_stroma  inflammatory regions (>= 64 px) whose border is < 50% stroma
#   necro_tumor  necrosis regions (>= 64 px) whose border is < 10% tumor
#   roi          P(class in tile | another tile of the ROI has it / none does)
#
# Sources: fit ground truth (where rules come from), dev ground truth, dev U-Net argmax
# and dev WCSP (run r_control, the 185 dev tiles every sweep uses). Also writes the
# ROI-presence inputs for wcsp/solve/presence.py (same format as the neighbor rule):
#   configs/tables/bcss_roi_prior.json, configs/tables/bcss_roi_<split>.json
#
#   python experiments/bcss_wcsp/rules/rule_stats.py

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
from wcsp.labels import CONNECTIVE, DEAD, IGNORE, INFLAMMATORY, NEOPLASTIC, NUM_LABELS  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
TILE = re.compile(r"(.*)_MPP-0_(\d+)_(\d+)_size512$")
TABLES = Path("experiments/bcss_wcsp/configs/tables")
EIGHT = np.ones((3, 3))


def roi_of(tile_id):
    return TILE.match(tile_id).group(1)


def shares(labels):
    ok = labels != IGNORE
    if ok.sum() < 1000:
        return None
    return np.bincount(labels[ok], minlength=NUM_LABELS)[:NUM_LABELS] / ok.sum()


def mask_of(present):
    return int(sum(1 << k for k in range(NUM_LABELS) if present[k]))


def border_shares(labels, cls, other, min_area=64):
    """Per region of cls (>= min_area px, >= 10 labeled border px): share of its border that is `other`."""
    comp, n = ndimage.label(labels == cls, structure=EIGHT)
    out = []
    for r in range(1, n + 1):
        region = comp == r
        if region.sum() < min_area:
            continue
        ring = ndimage.binary_dilation(region) & ~region & (labels != IGNORE)
        if ring.sum() >= 10:
            out.append(float((labels[ring] == other).mean()))
    return out


class Tally:
    def __init__(self, prior):
        self.prior = prior                      # (2^K,) P(set) from fit truth, or None
        self.tiles = 0
        self.rare = 0
        self.present = np.zeros(NUM_LABELS)
        self.any_px = np.zeros(NUM_LABELS)
        self.tiny = np.zeros(NUM_LABELS)
        self.pieces = np.zeros(NUM_LABELS)
        self.px = np.zeros(NUM_LABELS)
        self.small_px = np.zeros(NUM_LABELS)
        self.infl = []
        self.necro = []
        self.roi = defaultdict(lambda: np.zeros((2, NUM_LABELS)))

    def add(self, labels):
        s = shares(labels)
        if s is None:
            return None
        self.tiles += 1
        present = s >= 0.01
        self.present += present
        self.any_px += s > 0
        self.tiny += (s > 0) & (s < 0.02)
        if self.prior is not None:
            self.rare += self.prior[mask_of(present)] < 0.01
        for k in range(NUM_LABELS):
            comp, n = ndimage.label(labels == k, structure=EIGHT)
            if n:
                sizes = np.bincount(comp.ravel())[1:]
                self.pieces[k] += n
                self.px[k] += sizes.sum()
                self.small_px[k] += sizes[sizes < 256].sum()
        self.infl += border_shares(labels, INFLAMMATORY, CONNECTIVE)
        self.necro += border_shares(labels, DEAD, NEOPLASTIC)
        return present

    def add_roi(self, present, others):
        """present: (K,) bool for this tile; others: bitmask of classes in the ROI's other tiles."""
        for k in range(NUM_LABELS):
            key = "has" if others >> k & 1 else "none"
            self.roi[key][0, k] += 1
            self.roi[key][1, k] += present[k]

    def report(self):
        t = max(self.tiles, 1)
        infl, necro = np.array(self.infl), np.array(self.necro)
        out = {
            "tiles": self.tiles,
            "presence_rare_set_share_of_tiles": self.rare / t if self.prior is not None else None,
            "per_class_share_of_tiles_present": dict(zip(NAMES, (self.present / t).round(4).tolist())),
            "area_share_present_but_under_2pct": dict(zip(NAMES, (self.tiny / np.maximum(self.any_px, 1)).round(4).tolist())),
            "fragments_pieces_per_tile_with_class": dict(zip(NAMES, (self.pieces / np.maximum(self.any_px, 1)).round(2).tolist())),
            "fragments_share_px_in_pieces_under_256": dict(zip(NAMES, (self.small_px / np.maximum(self.px, 1)).round(4).tolist())),
            "infl_stroma_regions": len(infl),
            "infl_stroma_violation_share(<50%_stroma)": float((infl < 0.5).mean()) if len(infl) else None,
            "necro_tumor_regions": len(necro),
            "necro_tumor_violation_share(<10%_tumor)": float((necro < 0.1).mean()) if len(necro) else None,
            "necro_tumor_median_tumor_share": float(np.median(necro)) if len(necro) else None,
        }
        if self.roi:
            out["roi_P(class|other_ROI_tiles_have)"] = {
                key: dict(zip(NAMES, (v[1] / np.maximum(v[0], 1)).round(4).tolist())) for key, v in self.roi.items()}
        return out


def roi_masks(masks):
    """tile id -> OR of the masks of the other tiles of its ROI (-1 if it is alone)."""
    by_roi = defaultdict(list)
    for tile_id in masks:
        by_roi[roi_of(tile_id)].append(tile_id)
    out = {}
    for tiles in by_roi.values():
        for tile_id in tiles:
            others = [masks[o] for o in tiles if o != tile_id and masks[o] is not None]
            out[tile_id] = int(np.bitwise_or.reduce(others)) if others else -1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_rules.yaml")
    parser.add_argument("--run", default="r_control")
    parser.add_argument("--out", default="experiments/bcss_wcsp/rules/rule_stats.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prior_cost = json.loads(Path(cfg.solver["inner"]["prior"]).read_text())["cost"]
    prior = np.zeros(2 ** NUM_LABELS)
    for m, v in prior_cost.items():
        prior[int(m)] = np.exp(-v)
    out = {}

    # fit ground truth
    fit = pipeline.dataset_for(cfg, "fit")
    tally, masks = Tally(prior), {}
    for i in range(len(fit)):
        p = tally.add(fit[i].grid_target)
        masks[fit.ids[i]] = mask_of(p) if p is not None else None
    others = roi_masks(masks)
    for tile_id, m in masks.items():
        if m is not None and others[tile_id] >= 0:
            tally.add_roi([m >> k & 1 for k in range(NUM_LABELS)], others[tile_id])
    out["fit_truth"] = tally.report()
    roi = out["fit_truth"]["roi_P(class|other_ROI_tiles_have)"]
    (TABLES / "bcss_roi_prior.json").write_text(json.dumps(
        {"has": [roi["has"][n] for n in NAMES], "none": [roi["none"][n] for n in NAMES],
         "source": "fit ground truth, rules/rule_stats.py: P(class in tile | another tile of the ROI has it / none does)"},
        indent=2))

    # U-Net ROI context (all cached tiles), for the solver and the dev ROI tallies
    unet_others = {}
    for split in ("dev", "val"):
        store = pipeline.store_for(cfg, split)
        um = {}
        for tile_id in store.ids():
            s = np.bincount(store.load(tile_id).probs.argmax(0).ravel(), minlength=NUM_LABELS)[:NUM_LABELS]
            um[tile_id] = mask_of(s / s.sum() >= 0.01)
        unet_others[split] = roi_masks(um)
        (TABLES / f"bcss_roi_{split}.json").write_text(json.dumps(unet_others[split]))

    # dev: truth vs U-Net vs WCSP on the sweep tiles
    dev = pipeline.dataset_for(cfg, "dev")
    tallies = {name: Tally(prior) for name in ("truth", "unet", "wcsp")}
    for shard in sorted((cfg.runs_root / args.run / "dev").glob("shard_*.npz")):
        z = np.load(shard)
        for tile_id, pred, refined in zip(z["ids"].tolist(), z["argmax"], z["wcsp"]):
            target = dev[dev.index(tile_id)].grid_target
            ignored = target == IGNORE
            ctx = unet_others["dev"].get(tile_id, -1)
            for name, labels in (("truth", target), ("unet", np.where(ignored, IGNORE, pred)),
                                 ("wcsp", np.where(ignored, IGNORE, refined))):
                p = tallies[name].add(labels)
                if p is not None and ctx >= 0:
                    tallies[name].add_roi(p, ctx)
    for name, t in tallies.items():
        out[f"dev_{name}"] = t.report()
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
