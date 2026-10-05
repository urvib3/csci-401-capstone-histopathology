# Sanity checks and summary statistics for a downloaded BCSS variant.
#
#   python3 scripts/inspect_bcss.py --variant 224 --split train --limit 2000
#   python3 scripts/inspect_bcss.py --variant 512 --coarse --stitch
#
# Prints tile/ROI counts, the raw mask values that actually occur, class
# pixel frequencies and the 4-neighbor class adjacency matrix. --stitch
# writes the largest ROI, reassembled from its tiles, to disk.

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bcss import BCSSDataset, coarse_lut  # noqa: E402
from bcss.dataset import load_mask  # noqa: E402
from bcss.stats import adjacency_counts, class_counts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Inspect a BCSS variant.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--variant", choices=["224", "512"], default="224")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000,
                        help="number of masks to scan (0 = all)")
    parser.add_argument("--coarse", action="store_true",
                        help="512 only: map the 22 codes to the 5 coarse groups")
    parser.add_argument("--stitch", action="store_true")
    parser.add_argument("--out", default="bcss_inspect")
    args = parser.parse_args()

    lut, names = (coarse_lut() if args.coarse else (None, None))
    data = BCSSDataset(args.root, args.variant, args.split, lut=lut)
    rois = data.rois()
    print(f"{len(data)} tiles from {len(rois)} ROIs in {args.variant}/{args.split}")

    labeled = [i for i, t in enumerate(data.tiles) if t.mask_path is not None]
    if args.limit and len(labeled) > args.limit:
        rng = np.random.default_rng(0)
        labeled = rng.choice(labeled, args.limit, replace=False).tolist()
    masks = [load_mask(data.tiles[i].mask_path) for i in labeled]
    if lut is not None:
        masks = [lut[m] for m in masks]
    if not masks:
        print("No masks in this split.")
    else:
        values, counts = np.unique(np.concatenate([m.ravel() for m in masks]),
                                   return_counts=True)
        print(f"Mask values over {len(masks)} masks (value: fraction):")
        for v, c in zip(values, counts):
            label = f" {names[v]}" if names and v < len(names) else ""
            print(f"  {v:3d}{label}: {c / counts.sum():.4f}")

        labeled_values = values[values != 255]
        k = len(names) if names else int(labeled_values.max(initial=0)) + 1
        print("Class pixel counts:", class_counts(masks, k).tolist())
        adj = sum(adjacency_counts(m, k) for m in masks)
        print("4-neighbor adjacency counts (row/col = class):")
        print(adj)

    if args.stitch:
        os.makedirs(args.out, exist_ok=True)
        roi_id = max(rois, key=lambda r: len(rois[r]))
        image, mask = data.stitch_roi(roi_id)
        Image.fromarray(image).save(os.path.join(args.out, f"{roi_id}.png"))
        if mask is not None:
            Image.fromarray(mask).save(os.path.join(args.out, f"{roi_id}_mask.png"))
        print(f"Stitched {roi_id} ({image.shape[1]}x{image.shape[0]}) into {args.out}/")


if __name__ == "__main__":
    main()
