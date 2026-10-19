# PanNuke adapter (one fold of the official release).
#
# <root>/fold<k>/images.npy  (N, 256, 256, 3) float64 RGB
# <root>/fold<k>/masks.npy   (N, 256, 256, 6): channels 0-4 hold nucleus
#                            instance ids for neoplastic, inflammatory,
#                            connective, dead, non-neoplastic epithelial;
#                            channel 5 is background
# <root>/fold<k>/types.npy   (N,) tissue names
#
# Only nucleus pixels have a cell label; background pixels are IGNORE in the
# target. Splits of a fold: "dev" = every 5th patch (tuning), "test" = the
# rest, "all" = everything.

import os

import numpy as np

from ..labels import IGNORE, NUM_LABELS
from .base import Sample, SampleList

DEV_EVERY = 5


def default_root():
    return os.environ.get("PANNUKE_ROOT", os.path.join("data", "pannuke"))


def truth_from_mask(mask):
    """
    (256, 256, 6) PanNuke mask -> per-pixel labels and one instance map.

    Where class channels overlap, the later channel wins (same rule as
    benchmarks/pannuke/scripts/evaluate_ground_truth.py).
    """
    labels = np.full(mask.shape[:2], IGNORE, dtype=np.uint8)
    instances = np.zeros(mask.shape[:2], dtype=np.int32)
    count = 0
    for channel in range(NUM_LABELS):
        ids = mask[..., channel].astype(np.int64)
        for nucleus in np.unique(ids[ids > 0]):
            count += 1
            pixels = ids == nucleus
            labels[pixels] = channel
            instances[pixels] = count
    # Relabel so overwritten nuclei leave no gaps.
    kept = np.unique(instances[instances > 0])
    relabel = np.zeros(count + 1, dtype=np.int32)
    relabel[kept] = np.arange(1, len(kept) + 1)
    return labels, relabel[instances]


class PanNuke(SampleList):
    name = "pannuke"

    def __init__(self, split="dev", root=None, fold=2):
        if split not in ("dev", "test", "all"):
            raise ValueError("PanNuke split must be dev, test or all.")
        folder = os.path.join(root or default_root(), f"fold{fold}")
        self.images = np.load(os.path.join(folder, "images.npy"), mmap_mode="r")
        self.masks = np.load(os.path.join(folder, "masks.npy"), mmap_mode="r")
        self.tissues = np.load(os.path.join(folder, "types.npy"))
        self.split, self.fold = split, fold
        everything = range(len(self.images))
        if split == "dev":
            self.positions = [i for i in everything if i % DEV_EVERY == 0]
        elif split == "test":
            self.positions = [i for i in everything if i % DEV_EVERY != 0]
        else:
            self.positions = list(everything)
        self.ids = [f"fold{fold}_{i:04d}" for i in self.positions]

    def __getitem__(self, i):
        index = self.positions[i]
        image = np.clip(np.asarray(self.images[index]), 0, 255).astype(np.uint8)
        labels, instances = truth_from_mask(np.asarray(self.masks[index]))
        return Sample(
            id=self.ids[i],
            image=image,
            target=labels,
            truth_instances=instances,
            meta={"index": index, "tissue": str(self.tissues[index])},
        )
