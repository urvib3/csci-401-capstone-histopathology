# BCSS_512 adapter.
#
# Splits:
#   fit  train tiles outside the dev ROIs (cost statistics, calibration)
#   dev  every 8th train ROI, sorted by id (tuning; same rule as
#        benchmarks/bcss/scripts/common.py)
#   val  held-out hospitals (final numbers only)
#
# The grid is the central 256 x 256 of each 512 tile, which is the part the
# pretrained TIAToolbox U-Net predicts (benchmarks/bcss/scripts/evaluate_pretrained.py).

from benchmarks.bcss.loader import BCSSDataset, coarse_lut

from ..labels import BCSS_TO_WCSP, remap_labels
from .base import Sample, SampleList

TILE = 512
GRID = 256
OFFSET = (TILE - GRID) // 2
WINDOW = (slice(OFFSET, OFFSET + GRID), slice(OFFSET, OFFSET + GRID))
DEV_EVERY = 8


def _roi(name):
    return name.split("_MPP")[0]


class BCSS(SampleList):
    name = "bcss"

    def __init__(self, split="dev", root=None):
        if split not in ("fit", "dev", "val"):
            raise ValueError("BCSS split must be fit, dev or val.")
        lut, _ = coarse_lut()
        source = "val" if split == "val" else "train"
        self.data = BCSSDataset(root, variant="512", split=source, lut=lut)
        self.split = split
        positions = list(range(len(self.data.tiles)))
        if split != "val":
            rois = sorted({_roi(t.name) for t in self.data.tiles})
            held = set(rois[::DEV_EVERY])
            want_dev = split == "dev"
            positions = [i for i in positions if (_roi(self.data.tiles[i].name) in held) == want_dev]
        self.positions = positions
        self.ids = [self.data.tiles[i].name.removesuffix(".png") for i in positions]

    def __getitem__(self, i):
        tile = self.data[self.positions[i]]
        return Sample(
            id=self.ids[i],
            image=tile.image,
            target=remap_labels(tile.mask, BCSS_TO_WCSP),
            window=WINDOW,
            meta={"roi": tile.info.roi_id, "hospital": tile.info.slide_id.split("-")[1]},
        )
