"""Fetch 0.25 MPP ROIs using the official BCSS metadata, then tile at 512px.

Uses PathologyDataScience/BCSS roiBounds.csv and the public HistomicsTK API.
Retains full masks (codes 0 and 7 are ignored later), drops partial boundary
patches, and holds out hospitals OL/LL/E2/EW/GM/S3 as in the existing loader.
No credentials are needed for the public image and annotation endpoints.
"""
import argparse
import csv
import io
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1_000_000_000
API = "https://demo.kitware.com/histomicstk/api/v1"
META = "https://raw.githubusercontent.com/PathologyDataScience/BCSS/master/meta/roiBounds.csv"
HELD = {"OL", "LL", "E2", "EW", "GM", "S3"}


def fetch(url):
    for attempt in range(7):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return response.read()
        except Exception:
            if attempt == 6:
                raise RuntimeError(f"Failed to download {url}")
            time.sleep(2**attempt)


def label_image(mask):
    """Preserve both spatial axes for grayscale masks, including I;16 PNGs."""
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2 or values.max() > 21:
        raise ValueError(f"unexpected BCSS mask format: {mask.mode}, {values.shape}")
    return Image.fromarray(values.astype(np.uint8))


def prepare(row, slides, out):
    name = row[""]
    xmin, ymin, xmax, ymax = [int(float(row[k])) for k in ("xmin", "ymin", "xmax", "ymax")]
    roi = f"{name}_xmin{xmin}_ymin{ymin}"
    marker = out / "manifests" / f"{roi}.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("preparation_version") == 2:
            return previous
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    rgb_path, mask_path = raw / f"{roi}_rgb.png", raw / f"{roi}_mask.png"
    item = slides[name[:12]]
    url = (f"{API}/item/{item}/tiles/region?left={xmin}&right={xmax}&top={ymin}&bottom={ymax}"
           "&mm_x=0.00025&mm_y=0.00025&encoding=PNG")
    for path, source in ((rgb_path, url), (mask_path, row["mask_link"])):
        if not path.exists():
            tmp = path.with_suffix(".partial")
            tmp.write_bytes(fetch(source))
            # Validate before committing a possibly incomplete HTTP body.
            with Image.open(tmp) as image:
                image.verify()
            tmp.replace(path)
    image = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path)
    mask = label_image(mask)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    split = "val" if name.split("-")[1] in HELD else "train"
    image_dir, mask_dir = [out / "BCSS_512" / f"{split}{suffix}_512" for suffix in ("", "_mask")]
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for y in range(0, image.height-511, 512):
        for x in range(0, image.width-511, 512):
            filename = f"{roi}_MPP-0_{y}_{x}_size512.png"
            if not (image_dir / filename).exists():
                image.crop((x, y, x+512, y+512)).save(image_dir / filename)
            mask.crop((x, y, x+512, y+512)).save(mask_dir / filename)
            count += 1
    result = {"preparation_version": 2, "roi": roi, "split": split, "tiles": count, "size": image.size,
              "image_url": url, "mask_url": row["mask_link"], "mpp": .25}
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps(result, indent=2))
    print(f"{roi}: {split}, {count} tiles", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/bcss")
    p.add_argument("--metadata", help="local official meta/roiBounds.csv")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="debug: first N ROIs")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = Path(args.metadata).read_text() if args.metadata else fetch(META).decode()
    (out / "roiBounds.csv").write_text(metadata)
    rows = list(csv.DictReader(io.StringIO(metadata)))
    if args.limit:
        rows = rows[:args.limit]
    items = json.loads(fetch(f"{API}/item?folderId=5bbdeba3e629140048d017bb&limit=1000000"))
    slides = {item["name"][:12]: item["_id"] for item in items}
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, row, slides, out): row[""] for row in rows}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(f"{futures[future]}: {error}")
                print(errors[-1], flush=True)
    if errors:
        (out / "download_errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError(f"{len(errors)} ROI downloads failed; rerun to resume. See download_errors.json")
    results.sort(key=lambda row: row["roi"])
    manifest = {"preparation_version": 2, "source": "https://github.com/PathologyDataScience/BCSS", "mpp": .25,
                "tile_size": 512, "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
                "held_hospitals": sorted(HELD), "rois": results}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Complete: {len(results)} ROIs, {sum(r['tiles'] for r in results)} tiles", flush=True)


if __name__ == "__main__":
    main()
