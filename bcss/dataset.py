# Loader for the Kaggle BCSS release (whats2000/breast-cancer-semantic-segmentation-bcss).
#
# Layout of the download (version 2):
#
#   <root>/BCSS/{train,val,test}/*.png           224x224 RGB tiles
#   <root>/BCSS/{train,val}_mask/*.png           masks, 3 merged labels (no test masks)
#   <root>/BCSS_512/{train,val}_512/*.png        512x512 RGB tiles
#   <root>/BCSS_512/{train,val}_mask_512/*.png   masks, original 22 BCSS codes
#
# A mask has the same file name as its image. File names look like
#
#   TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0_448_2240_size224.png
#
# i.e. <slide>_xmin<x>_ymin<y> identifies one annotated ROI of a TCGA slide
# (151 ROIs total), followed by the tile's pixel offset inside that ROI and
# the tile size. Tiles from one ROI can therefore be stitched back together.
#
# Tiles are cut on a fixed grid; partial tiles at the ROI edge were dropped.
#
# Split caveat: the 224 splits are random per tile, so every ROI has tiles
# in train, val and test. The 512 splits hold out whole TCGA tissue source
# sites (OL, LL, E2, EW, GM, S3 in val), giving 106 train / 45 val ROIs.

import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .classes import IGNORE_INDEX

VARIANTS = {
    "224": {
        "dir": "BCSS",
        "tile_size": 224,
        "splits": {
            "train": ("train", "train_mask"),
            "val": ("val", "val_mask"),
            "test": ("test", None),
        },
    },
    "512": {
        "dir": "BCSS_512",
        "tile_size": 512,
        "splits": {
            "train": ("train_512", "train_mask_512"),
            "val": ("val_512", "val_mask_512"),
        },
    },
}

_TILE_NAME = re.compile(
    r"^(?P<slide>TCGA-[A-Za-z0-9-]+?)_xmin(?P<xmin>\d+)_ymin(?P<ymin>\d+)"
    r"_MPP-\d+_(?P<off0>\d+)_(?P<off1>\d+)_size(?P<size>\d+)\.png$"
)


@dataclass(frozen=True)
class TileInfo:
    """
    Metadata for one tile, parsed from its file name.

    offset:
        (row, col) pixel offset of the tile's top-left corner inside its ROI
        (order confirmed by make_patches_ori.py, shipped with the dataset).
    """

    name: str
    split: str
    image_path: str
    mask_path: str | None
    slide_id: str
    roi_xmin: int
    roi_ymin: int
    offset: tuple[int, int]
    size: int

    @property
    def roi_id(self):
        return f"{self.slide_id}_xmin{self.roi_xmin}_ymin{self.roi_ymin}"


@dataclass
class Sample:
    image: np.ndarray  # (H, W, 3) uint8
    mask: np.ndarray | None  # (H, W) uint8, None for unlabeled tiles
    info: TileInfo


def parse_tile_name(name):
    """Return the regex groups of a BCSS tile file name, or raise ValueError."""
    match = _TILE_NAME.match(name)
    if match is None:
        raise ValueError(f"Unrecognized BCSS tile name: {name}")
    return match


def default_root():
    """Dataset root from $BCSS_ROOT, falling back to ./data/bcss."""
    return os.environ.get("BCSS_ROOT", os.path.join("data", "bcss"))


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8)


class BCSSDataset:
    """
    Indexable collection of BCSS tiles.

    root:
        Directory containing BCSS/ and/or BCSS_512/. Defaults to
        default_root().
    variant:
        "224" (3 merged labels) or "512" (22 original codes).
    split:
        A split name or a list of them, e.g. ["train", "val"].
    lut:
        Optional lookup table applied to every mask as lut[mask], e.g. from
        bcss.classes.coarse_lut(). Must have IGNORE_INDEX for don't-care
        codes.

    dataset[i] returns a Sample with numpy arrays.
    """

    def __init__(self, root=None, variant="224", split="train", lut=None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}.")
        spec = VARIANTS[variant]
        self.root = root or default_root()
        self.variant = variant
        self.tile_size = spec["tile_size"]
        self.splits = [split] if isinstance(split, str) else list(split)
        self.lut = None if lut is None else np.asarray(lut, dtype=np.uint8)

        self.tiles = []
        for name in self.splits:
            if name not in spec["splits"]:
                raise ValueError(
                    f"Split {name!r} not available for variant {variant}; "
                    f"choose from {sorted(spec['splits'])}."
                )
            image_dir, mask_dir = spec["splits"][name]
            self.tiles.extend(
                self._index_split(
                    name,
                    os.path.join(self.root, spec["dir"], image_dir),
                    mask_dir and os.path.join(self.root, spec["dir"], mask_dir),
                )
            )

    @staticmethod
    def _index_split(split, image_dir, mask_dir):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"{image_dir} not found. Download the dataset with "
                "scripts/download_bcss.py or set $BCSS_ROOT."
            )
        tiles = []
        for name in sorted(os.listdir(image_dir)):
            if not name.endswith(".png"):
                continue
            m = parse_tile_name(name)
            tiles.append(TileInfo(
                name=name,
                split=split,
                image_path=os.path.join(image_dir, name),
                mask_path=mask_dir and os.path.join(mask_dir, name),
                slide_id=m["slide"],
                roi_xmin=int(m["xmin"]),
                roi_ymin=int(m["ymin"]),
                offset=(int(m["off0"]), int(m["off1"])),
                size=int(m["size"]),
            ))
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        info = self.tiles[index]
        mask = None
        if info.mask_path is not None:
            mask = load_mask(info.mask_path)
            if self.lut is not None:
                mask = self.lut[mask]
        return Sample(load_image(info.image_path), mask, info)

    def masks(self):
        """Iterate over masks only, skipping image decoding."""
        for info in self.tiles:
            if info.mask_path is None:
                continue
            mask = load_mask(info.mask_path)
            yield mask if self.lut is None else self.lut[mask]

    def rois(self):
        """Map roi_id -> list of tile indices in this dataset."""
        groups = defaultdict(list)
        for index, info in enumerate(self.tiles):
            groups[info.roi_id].append(index)
        return dict(groups)

    def stitch_roi(self, roi_id):
        """
        Reassemble every tile of one ROI present in this dataset.

        Positions with no tile (e.g. tiles that fall in another split) are
        black in the image and IGNORE_INDEX in the mask.

        returns:
            image: (H, W, 3) uint8 mosaic
            mask:  (H, W) uint8 mosaic, or None if no tile has a mask
        """
        indices = self.rois().get(roi_id)
        if not indices:
            raise KeyError(f"No tiles for ROI {roi_id} in splits {self.splits}.")
        s = self.tile_size
        height = max(self.tiles[i].offset[0] for i in indices) + s
        width = max(self.tiles[i].offset[1] for i in indices) + s

        image = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.full((height, width), IGNORE_INDEX, dtype=np.uint8)
        has_mask = False
        for i in indices:
            sample = self[i]
            r, c = sample.info.offset
            image[r:r + s, c:c + s] = sample.image
            if sample.mask is not None:
                mask[r:r + s, c:c + s] = sample.mask
                has_mask = True
        return image, (mask if has_mask else None)
