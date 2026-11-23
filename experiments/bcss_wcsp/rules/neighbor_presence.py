# Inputs for the neighbor-tile rule in wcsp/solve/presence.py.
#
#   configs/tables/bcss_neighbor_prior.json   P(class in tile | an adjacent tile has it / none does),
#                                             from fit ground truth (rules/check_rules.json)
#   configs/tables/bcss_neighbors_<split>.json  tile id -> bitmask of classes the U-Net predicts
#                                             (>= 1% of pixels) in any of the 4 adjacent tiles; -1 = none
#
#   python experiments/bcss_wcsp/rules/neighbor_presence.py --splits dev val

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.labels import NUM_LABELS  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
TILE = re.compile(r"(.*)_MPP-0_(\d+)_(\d+)_size512$")
TABLES = Path("experiments/bcss_wcsp/configs/tables")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--splits", nargs="+", default=["dev", "val"])
    args = parser.parse_args()

    fit = json.loads(Path("experiments/bcss_wcsp/rules/check_rules.json").read_text())["fit_truth"]["R1_P(class in tile)"]
    prior = {"has": [fit["neighbor_has"][n] for n in NAMES], "none": [fit["no_neighbor_has"][n] for n in NAMES],
             "source": "fit ground truth, rules/check_rules.py"}
    (TABLES / "bcss_neighbor_prior.json").write_text(json.dumps(prior, indent=2))

    cfg = load_config(args.config)
    for split in args.splits:
        store = pipeline.store_for(cfg, split)
        masks = {}
        for tile_id in store.ids():
            labels = store.load(tile_id).probs.argmax(0)
            share = np.bincount(labels.ravel(), minlength=NUM_LABELS)[:NUM_LABELS] / labels.size
            masks[tile_id] = int(sum(1 << k for k in range(NUM_LABELS) if share[k] >= 0.01))
        index = {}
        for tile_id in masks:
            roi, a, b = TILE.match(tile_id).groups()
            index[(roi, int(a), int(b))] = tile_id
        out = {}
        for (roi, a, b), tile_id in index.items():
            nbrs = [index.get((roi, a + da, b + db)) for da, db in ((512, 0), (-512, 0), (0, 512), (0, -512))]
            nbrs = [masks[n] for n in nbrs if n]
            out[tile_id] = int(np.bitwise_or.reduce(nbrs)) if nbrs else -1
        (TABLES / f"bcss_neighbors_{split}.json").write_text(json.dumps(out))
        print(split, len(out), "tiles,", sum(v >= 0 for v in out.values()), "with neighbors")


if __name__ == "__main__":
    main()
