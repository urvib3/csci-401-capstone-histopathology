# wcsp — pixel-level WCSP for BCSS

Separate from the PanNuke Deliverable 4 scripts in `benchmarks/deliverable_4/`.
This package is the **BCSS** pipeline: every pixel of a tile's central
256 × 256 is a variable over 5 labels, unary costs come from the pretrained
TIAToolbox U-Net's per-pixel softmax, and pairwise costs follow the
handwritten notes (cost table over 5 neighbor types, divided by distance^η).

Label order everywhere: neoplastic, inflammatory, connective, dead, healthy.
BCSS (tumor, stroma, inflammatory, necrosis, other) is remapped on load;
BCSS "other" sits in the healthy slot, and the `other_neutral` table drops
the healthy-tissue rules for it.

| Module | What it does |
| --- | --- |
| `labels.py` | label order, BCSS remapping, colours |
| `model/` | neighbor types, cost tables, unary (−log p), image terms, `build_problem` |
| `solve/` | `energy`, `condition` (fix pixels, fold their costs in), argmax, ICM, toulbar2, the 100/56/100 block scheme |
| `data/bcss.py` | BCSS_512 splits: fit / dev (held-out train ROIs) / val (held-out hospitals) |
| `classifiers/` | TIAToolbox U-Net wrapper, probability cache, temperature calibration |
| `evaluate/` | mIoU / Dice, rule-violation counts, statistical cost table |
| `pipeline.py`, `cli.py` | `python -m wcsp predict / calibrate / solve / summarize / elicit / show` |

`data/pannuke.py` and `evaluate/nuclei.py` are an untested start on PanNuke
support and are not used by the BCSS pipeline.

```bash
python -m pytest tests/wcsp          # numpy-only tests, no data needed
sbatch --array=0-19 slurm/wcsp.sbatch predict --config configs/bcss_unet.yaml --split dev --shards 20
sbatch --array=0-19 slurm/wcsp.sbatch solve   --config configs/bcss_unet.yaml --split dev --shards 20
python -m wcsp summarize --config configs/bcss_unet.yaml --split dev
```
