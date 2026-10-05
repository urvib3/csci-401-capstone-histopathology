# histo-wcsp

AI methods for histopathology: labeling tissue regions (tumor, stroma, healthy, ...)
using Weighted Constraint Satisfaction Problems (WCSP) and related combinatorial
optimization techniques.

## Layout

- `sample_problems/` - warm-up problems cast as WCSPs (domain sizes, arities,
  treewidth) and solved with Toulbar2, plus notes on Local Search / A* alternatives.
- `bcss/` - loader and WCSP-oriented primitives for the BCSS breast cancer
  segmentation benchmark (see below).
- `scripts/` - dataset download and inspection.
- `tests/` - `python3 -m pytest tests` (runs without the dataset).
- `docs/design/` - design docs: [segmentation + WCSP pipeline](docs/design/segmentation-wcsp-pipeline.md)
  (proposed) and [segmentation algorithm reference](docs/design/segmentation-algorithms.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## BCSS dataset

We use the Kaggle release of BCSS
([whats2000/breast-cancer-semantic-segmentation-bcss](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss),
CC0, ~8.3 GB), tiled from the 151 annotated TCGA-BRCA ROIs of
[PathologyDataScience/BCSS](https://github.com/PathologyDataScience/BCSS).
Cite Amgad et al., *Bioinformatics* 2019, doi:10.1093/bioinformatics/btz083.

```bash
python3 scripts/download_bcss.py   # no Kaggle login needed; -> data/bcss (or set $BCSS_ROOT)
python3 scripts/inspect_bcss.py --variant 512 --coarse --stitch
```

| Variant | Splits (tiles) | Mask labels | Split unit |
| --- | --- | --- | --- |
| `224` (`BCSS/`) | train 30760, val 5429, test 4021 (**test has no masks**) | 0 other, 1 tumor, 2 stroma | random per tile: all 151 ROIs appear in every split |
| `512` (`BCSS_512/`) | train 6000, val 2768 | 22 original BCSS codes | by hospital: val holds out TCGA sites OL, LL, E2, EW, GM, S3 (106 / 45 ROIs) |

The 512 variant is the cleaner benchmark: full labels and no ROI leakage
between train and val. Mask code 0 (`outside_roi`) and 7 (`exclude`) are
"don't care" and map to `IGNORE_INDEX` (255) under the provided LUTs. In the
224 variant, label 0 also absorbs `outside_roi`, so background is scored as
"other" there.

Tile names encode `<slide>_xmin<x>_ymin<y>` (one of 151 annotated ROIs) and
the tile's `(row, col)` offset in that ROI, so `stitch_roi` can rebuild ROIs
up to ~9200 x 9200 px for region-level WCSPs.

```python
from bcss import BCSSDataset, CellGrid, coarse_lut

lut, names = coarse_lut()        # 22 codes -> tumor/stroma/inflammatory/necrosis/other
data = BCSSDataset(variant="512", split="train", lut=lut)
sample = data[0]                 # .image (512,512,3) uint8, .mask (512,512), .info

grid = CellGrid(512, 512, cell_size=16)          # 32 x 32 lattice = 1024 variables
truth = grid.majority_labels(sample.mask, len(names))
edges = grid.edges(connectivity=4)               # scopes for binary cost functions

image, mask = data.stitch_roi(sample.info.roi_id)  # reassemble a whole ROI
```

Primitives for building a WCSP:

- `bcss.grid.CellGrid` - cells as variables, lattice edges, per-cell label
  histograms / majority labels, pooling of per-pixel class probabilities,
  and upsampling a cell labeling back to pixels.
- `bcss.grid.unary_costs_from_probabilities`, `to_integer_costs` - `-log p`
  unary tables and integer scaling for Toulbar2.
- `bcss.stats` - class pixel counts and neighbor class co-occurrence, with
  `negative_log_costs` to turn either into cost tables.
- `bcss.metrics` - confusion matrix, pixel accuracy, per-class IoU / Dice,
  ignoring don't-care pixels.
- `bcss.torch_data` - optional PyTorch `Dataset`/`DataLoader` adapter.
