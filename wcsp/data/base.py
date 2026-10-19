# What every dataset adapter returns.
#
# A Sample carries the classifier's input image and, separately, the window
# of that image the WCSP labels (the "grid"). For BCSS the U-Net sees a
# 512 px tile but only predicts its central 256 px, so the grid is that
# centre; for PanNuke the grid is the whole 256 px patch.

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sample:
    id: str
    image: np.ndarray                    # (H, W, 3) uint8 classifier input
    target: np.ndarray | None            # (H, W) uint8 wcsp labels, IGNORE = not scored
    window: tuple = (slice(None), slice(None))  # grid inside image
    truth_instances: np.ndarray | None = None   # (H, W) int32 nucleus ids (PanNuke)
    meta: dict = field(default_factory=dict)

    @property
    def grid_image(self):
        return self.image[self.window]

    @property
    def grid_target(self):
        return None if self.target is None else self.target[self.window]

    @property
    def grid_truth_instances(self):
        return None if self.truth_instances is None else self.truth_instances[self.window]


class SampleList:
    """Shared indexing / sharding for the adapters."""

    name = "dataset"

    def __len__(self):
        return len(self.ids)

    def index(self, sample_id):
        if not hasattr(self, "_positions"):
            self._positions = {s: i for i, s in enumerate(self.ids)}
        return self._positions[sample_id]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        raise NotImplementedError


def parse_shard(text):
    """'3/20' -> (3, 20); None -> (0, 1)."""
    if not text:
        return 0, 1
    index, count = (int(v) for v in text.split("/"))
    return index, count


def select(size, shard=(0, 1), limit=0):
    """
    Positions a job works on: at most `limit` samples spread evenly over the
    split (0 = all), then every count-th of those for shard (index, count).
    The same limit gives the same subset whatever the shard count.
    """
    index, count = shard
    if not 0 <= index < count:
        raise ValueError(f"shard {index} is outside 0..{count - 1}.")
    positions = np.arange(size)
    if limit and limit < size:
        positions = np.linspace(0, size - 1, limit).round().astype(int)
    return positions[index::count].tolist()
