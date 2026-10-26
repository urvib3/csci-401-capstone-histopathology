# wcsp — pixel-level WCSP for BCSS

Separate from the PanNuke experiments in `experiments/pannuke_d4/` and `experiments/pannuke_d5/`. BCSS experiment configs, SLURM scripts and figures live in `experiments/bcss_wcsp/`.
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

## Results

Full results (λ sweeps, handwritten vs. data-derived b, image term, ICM vs. toulbar2, figures)
are in [Deliverable 5, BCSS section](../deliverables/deliverable_5.md#bcss-pixel-level-wcsp-on-a-u-net-urvi).
Headline on 2,124 held-out-hospital val tiles: U-Net mIoU 0.6489 → WCSP 0.6507
(+0.0019, 95% CI [+0.0010, +0.0027]); pixel accuracy 0.8414 → 0.8419.

## Experiment: native-resolution U-Net

**Question:** the U-Net predicts at half resolution and its output is
upsampled ×2 (bilinear), so neighboring pixels share predictions, at exactly
the 1–2 px scale the WCSP rules act on. Does a genuinely per-pixel U-Net
output change the U-Net's accuracy, or how much the WCSP adds?

**Native output without retraining (shift-and-stitch):** run the network on
the tile shifted by 0 or 1 px in each direction and interleave the four
half-resolution outputs, so every pixel gets its own prediction (4× the
U-Net compute; the WCSP solve is unchanged). Classifier `tia_unet_native`,
config `configs/bcss_unet_native.yaml`; everything else is identical to
`configs/bcss_unet.yaml`, so the two runs are paired on the same tiles.

**Expected size of the effect:** small for the U-Net alone. Rounding the
ground truth itself through half resolution and back costs only 0.001 mIoU
on val, and a 100-tile dev prototype gave native 0.708 vs bilinear 0.711
mIoU, with 98% of pixels labeled the same.

**Protocol:**

1. Cache both classifiers' probabilities on dev and val.
2. λ sweep (0.5–8) for each classifier on the same 200 dev tiles.
3. For each classifier, pick the λ with the best WCSP mIoU on dev.
4. Run both on all val tiles at their λ, and compare with a paired
   ROI bootstrap: U-Net native vs bilinear, WCSP native vs bilinear, and
   the difference in WCSP gain, (WCSP − U-Net) native minus bilinear.

```bash
for c in bcss_unet bcss_unet_native; do
  for s in dev val; do sbatch --array=0-19 slurm/wcsp.sbatch predict --config configs/$c.yaml --split $s --shards 20; done
done
sbatch --array=0-4 slurm/sweep.sbatch configs/bcss_unet.yaml        configs/sweeps/bcss_dev.txt        --split dev --limit 200
sbatch --array=0-4 slurm/sweep.sbatch configs/bcss_unet_native.yaml configs/sweeps/bcss_dev_native.txt --split dev --limit 200
# then, with each config's best dev lambda (L, LN):
sbatch --array=0-19 slurm/wcsp.sbatch solve --config configs/bcss_unet.yaml        --split val --shards 20 --run val_bilinear --set problem.lam=L
sbatch --array=0-19 slurm/wcsp.sbatch solve --config configs/bcss_unet_native.yaml --split val --shards 20 --run val_native   --set problem.lam=LN
python -m wcsp summarize --config configs/bcss_unet.yaml        --split val --run val_bilinear
python -m wcsp summarize --config configs/bcss_unet_native.yaml --split val --run val_native
python -m wcsp compare   --config configs/bcss_unet.yaml        --split val --run val_native --against val_bilinear
```

`compare` writes `compare_vs_<baseline>.json` into the compared run.

**Trained full-resolution level (third arm):** `wcsp/classifiers/fullres_unet.py`
adds one encoder level (2 convs on the raw image) and one decoder level to the
pretrained U-Net, predicting a correction to its upsampled logits. The new
layers start at zero, so before training the model reproduces the pretrained
U-Net. Train them on fit, picking the epoch on dev, then use
`configs/bcss_unet_fullres.yaml` and `configs/sweeps/bcss_dev_fullres.txt`
exactly like the native arm:

```bash
python benchmarks/bcss/scripts/train_fullres_unet.py --epochs 5   # writes data/outputs/fullres_unet/best.pt
```

## Caveats

- BCSS "other" (fat, vessels, normal ducts, …) is not really "healthy", but it
  sits in the healthy slot of the cost table.
- One pixel is not one cell here (0.25 µm/px); the rules act on tissue
  regions, not individual cells.
- toulbar2 is supported, but with 20 s per block it finds much worse solutions
  than ICM on these 10k–25k-variable blocks, so ICM is the default.

`data/pannuke.py` and `evaluate/nuclei.py` are an untested start on PanNuke
support and are not used by the BCSS pipeline.

```bash
python -m pytest tests/wcsp          # numpy-only tests, no data needed
sbatch --array=0-19 experiments/bcss_wcsp/slurm/wcsp.sbatch predict --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split dev --shards 20
sbatch --array=0-19 experiments/bcss_wcsp/slurm/wcsp.sbatch solve   --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split dev --shards 20
python -m wcsp summarize --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split dev
```
