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

## What is actually happening (plain-language version)

**The starting point.** A U-Net is a neural network that looks at a 512 × 512
tissue image and, for every pixel in the middle 256 × 256, outputs five
probabilities: how likely that pixel is tumor, inflammatory, stroma, necrosis
or "other". We use a pretrained one from TIAToolbox exactly as it comes (no
training, no calibration). Taking the most likely class at each pixel
("argmax") is our baseline.

**The problem with the baseline.** The U-Net decides each pixel on its own
evidence. It has no explicit idea of biology such as "dead tissue does not sit
in the middle of healthy tissue" or "neighboring pixels are usually the same
tissue".

**What the WCSP adds.** We treat labeling the image as one big puzzle with
65,536 pieces (one per pixel), each taking one of 5 labels. Every possible
labeling gets a total cost, and we search for a cheap one:

- **Unary cost (trust the U-Net):** for each pixel, −log(probability the U-Net
  gave that label). Labels the U-Net likes are cheap; labels it thinks are
  unlikely are expensive.
- **Pairwise cost (the biology rules):** for every pair of pixels within
  2√2 px of each other, a penalty from the professor's table when their labels
  "shouldn't" touch (e.g. dead next to healthy = 10, tumor next to healthy = 9,
  same label = 0), divided by their distance.
- **λ** sets how much the rules count compared with the U-Net. λ = 0 gives
  back the U-Net exactly; a large λ lets the rules overrule it.

**How we search.** Trying every labeling is impossible (5^65,536). We use the
notes' block strategy: fix the 56-px cross through the middle of the image,
improve each 100 × 100 corner, then fix the corners and improve the cross, and
repeat. Inside each block, a greedy solver (ICM) changes one pixel at a time
whenever that lowers the total cost. The cost can only go down, so the result
is never worse than the U-Net *by the cost's own measure*; whether it is more
*accurate* is what we test against the pathologists' labels.

## Results so far

**First check: 10 val tiles** (hospitals the U-Net never saw; evenly spread
over the val set). Pixel accuracy and mIoU (mean overlap between predicted
and true regions over the 5 classes):

| λ | Method | Pixel accuracy | mIoU |
| --- | --- | ---: | ---: |
| 1 | U-Net alone | 0.798 | 0.482 |
| 1 | U-Net + WCSP | 0.800 | 0.483 |
| 4 | U-Net alone | 0.798 | 0.482 |
| 4 | U-Net + WCSP | 0.800 | 0.483 |

**What this means:**

- The WCSP helps, but only a little (+0.2 points accuracy). Gains come mostly
  from tumor and stroma pixels.
- **Why so little:** the U-Net's output is already smooth, with big blobs of
  one class. The WCSP can only move pixels near the edges between regions, so
  it changes well under 1% of pixels. It cannot fix a whole region the U-Net
  got wrong, because the U-Net is confident there and the rules only look 2–3
  pixels away.
- **Ten tiles is not enough to trust the difference.** "Other/healthy" scores 0
  for both methods here, which drags mIoU down (the full-val U-Net mIoU is
  0.649).

**Running now:** λ ∈ {0.5, 1, 2, 4, 8} on 200 dev tiles (tuning set) to pick
λ, then U-Net vs. U-Net + WCSP on all 2,768 val tiles, with a 95% confidence
interval from resampling whole tissue regions (ROIs).

## Caveats

- BCSS "other" (fat, vessels, normal ducts, …) is not really "healthy", but it
  sits in the healthy slot of the cost table.
- One pixel is not one cell here (0.25 µm/px); the rules act on tissue
  regions, not individual cells.
- Exact solving with toulbar2 is supported, but on a 100 × 100 block it did not
  reliably beat the greedy solver within 10–20 s, so the greedy solver is the
  default.

`data/pannuke.py` and `evaluate/nuclei.py` are an untested start on PanNuke
support and are not used by the BCSS pipeline.

```bash
python -m pytest tests/wcsp          # numpy-only tests, no data needed
sbatch --array=0-19 slurm/wcsp.sbatch predict --config configs/bcss_unet.yaml --split dev --shards 20
sbatch --array=0-19 slurm/wcsp.sbatch solve   --config configs/bcss_unet.yaml --split dev --shards 20
python -m wcsp summarize --config configs/bcss_unet.yaml --split dev
```
