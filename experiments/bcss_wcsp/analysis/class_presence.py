# How many WCSP errors come from a class predicted in a tile where the ground truth
# does not have it at all (< 1% of the tile's labeled pixels)? Those are tile-level
# mistakes that only a rule about which classes co-occur could catch.
#
# Ceiling: give those pixels their ground-truth label (an oracle presence rule);
# no tile-level presence term can do better than this.
#
#   python experiments/bcss_wcsp/analysis/class_presence.py --run valfull_data_lam8

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.evaluate.pixel import pixel_confusion, pixel_report  # noqa: E402
from wcsp.labels import IGNORE, NUM_LABELS  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--run", default="valfull_data_lam8")
    parser.add_argument("--out", default="experiments/bcss_wcsp/analysis/class_presence_val.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = pipeline.dataset_for(cfg, "val")

    errors = absent_errors = 0
    by_class = np.zeros(NUM_LABELS)
    confusion = {"wcsp": 0, "oracle_presence": 0}
    for shard in sorted((cfg.runs_root / args.run / "val").glob("shard_*.npz")):
        z = np.load(shard)
        for sample_id, refined in zip(z["ids"].tolist(), z["wcsp"]):
            target = data[data.index(sample_id)].grid_target
            labeled = target != IGNORE
            share = np.bincount(target[labeled], minlength=NUM_LABELS) / max(labeled.sum(), 1)
            absent = share < 0.01
            wrong = labeled & (refined != target)
            from_absent = wrong & absent[refined]
            errors += wrong.sum()
            absent_errors += from_absent.sum()
            by_class += np.bincount(refined[from_absent], minlength=NUM_LABELS)
            fixed = np.where(from_absent, target, refined)
            confusion["wcsp"] = confusion["wcsp"] + pixel_confusion(refined, target)
            confusion["oracle_presence"] = confusion["oracle_presence"] + pixel_confusion(fixed, target)

    report = {
        "run": args.run,
        "wcsp_errors": int(errors),
        "share_from_absent_class": float(absent_errors / errors),
        "absent_class_errors_by_predicted_class": {NAMES[k]: float(by_class[k] / absent_errors)
                                                   for k in range(NUM_LABELS)},
        "scores": {name: {k: v for k, v in pixel_report(c).items() if k in ("pixel_acc", "mIoU")}
                   for name, c in confusion.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
