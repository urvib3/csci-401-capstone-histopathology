BCSS overnight experiments
==========================

Use `nlp_hiprio`, account `jessetho_1732`. Everything lives under this
checkout on `/scratch2/rnene`; the new Python environment is `.venv`.

Data preparation
----------------

`scripts/prepare_bcss_upstream.py` reads the official
[BCSS metadata](https://github.com/PathologyDataScience/BCSS), downloads RGB
ROIs at 0.25 MPP and original masks, and makes nonoverlapping 512px tiles. Preparation version 2 preserves
both axes of the upstream 16-bit grayscale label masks.
Incomplete border tiles are dropped. Hospitals OL, LL, E2, EW, GM and S3
are held out as val, matching this repository's previous convention. Every
8th sorted training ROI is dev; all other training ROIs are fit. Codes 0 and
7 remain ignored. No val labels select models or parameters.

This is a fresh tiling of upstream data, **not a byte-identical copy of the
Kaggle benchmark**. Tile counts and pixels may differ from previous reports.
Do not interpret differences against published old numbers as treatment
changes. Compare the new arms on the same newly prepared dataset. The
manifest records ROI names, download URLs, MPP, dimensions and tile counts.
The server's requested MPP is recorded; the original BCSS downloader rounds
its mm query to four decimal places, whereas this script requests 0.00025mm.

```bash
.venv/bin/python scripts/prepare_bcss_upstream.py --workers 4
```

Downloads resume at the ROI level, retain raw files, and report any failures
in `data/bcss/download_errors.json`. Start training only after all 151 ROIs
are complete and `data/bcss/manifest.json` exists (preparation version 2).
The prepared data has 7,768 fit, 1,012 dev and 4,028 val tiles. Free space should cover
both raw ROIs and tiles, checkpoints, and cached probabilities.

Training arms
-------------

`benchmarks/bcss/scripts/train_experiments.py` fine-tunes the TIAToolbox
ResNet-50 U-Net with AdamW, augmentation, batch 4, 10 epochs, seed 0,
base learning rate 1e-4 and cosine decay. It retains epoch 0 if no epoch
improves dev mIoU. The full-resolution head uses learning rate 1e-3.

| Array index | Arm | Architecture and scored field | Loss |
| --- | --- | --- | --- |
| 0 | `ce` | Original half-resolution output, bilinear probabilities, central 256 | Cross-entropy |
| 1 | `full512` | Full-resolution encoder/decoder head, all 512 pixels per side | Cross-entropy |
| 2 | `pairwise_0.01` | Same as `ce` | CE + 0.01 × expected pairwise cost |
| 3 | `pairwise_0.1` | Same as `ce` | CE + 0.1 × expected pairwise cost |
| 4 | `pairwise_1` | Same as `ce` | CE + 1 × expected pairwise cost |

The full-resolution arm fine-tunes the entire backbone plus the new head;
it outputs learned logits for every image pixel in a single pass. Padding
supplies border context. Its dev log also reports center mIoU. Evaluation
configs `ce_full512` and `pretrained_full512` score the original architecture
over the same complete field, allowing comparisons on matching pixels.
This arm changes architecture and training field together; separating those
effects would require another full-field, original-architecture training arm.

The pairwise loss is `sum_(p,q) w_pq P_p^T b P_q / labeled_pixel_count`.
It uses all five local neighborhood types, handwritten b, eta=1, and the
same local normalization as WCSP. The loss remaps the table into BCSS channel
order. Pairs touching ignored pixels are masked. It is separate from the
full-resolution treatment, and no WCSP solve runs inside backpropagation.
The scalar training weight is independent of the inference WCSP lambda.

Checkpoints, settings and epoch logs: `data/outputs/tonight/<arm>/`.
Resume `last.pt` with identical settings; use a new output directory when
changing loss, resolution, seed or training schedule. A completed checkpoint
and an interrupted checkpoint both carry width and architecture metadata.

WCSP arms
---------

The WCSP sweeps use the same 200 evenly spaced dev tiles, raw
probabilities (T=1), lambda=8. The pretrained U-Net supplies a fresh baseline.
Training arms are evaluated with and without WCSP refinement. All training
weights and distant-edge settings are selected on dev; val is a later,
separate step once settings are frozen.

For distant relations, every unordered pair at Euclidean distance **> d**
is selected independently with probability p. The experiment uses d in
{8,32} and p in {1e-6,1e-5}, on the original central 256 grid, isolating the
change from retraining. Compare with `tonight_long_control`, which uses the
same whole-grid ICM solver with p=0. A 256 grid has about 2.15 billion unordered pairs,
so these probabilities produce about 2,100 or 21,000 candidate edges per
tile before distance filtering. For 512 grids, p=1e-5 gives about 344,000.

Sparse edges carry `lambda * b / distance^eta`, divided by the *existing*
local normalization constant. Adding distant edges does not dilute local
weights; there is no inverse-probability reweighting. Image/instance extra
terms remain local. Seed 0 gives the same sampled geometry for paired runs.
`long_max_edges` explicitly rejects overly dense requests rather than
silently changing p; d must be at least sqrt(8) to avoid duplicate local
edges. Distant-edge ICM uses extra graph colors. Blocks with distant edges
are conditioned sequentially because different blocks may interact.

The solver comparison uses the same 100/56/100 layout, four outer rounds,
and either ICM, Toulbar2 (2 seconds search per block), or ICM followed by
Toulbar2 within each block. Hybrid seeds Toulbar2's value order and upper
bound from ICM and retains ICM if search finds no improvement. The time
limit covers search, not Python construction. Compare measured end-to-end
time and final energy alongside mIoU; hybrid has extra ICM compute, so this
is not an equal-total-wall-time benchmark. It cannot promise better accuracy
or runtime in advance. Full-512 inference uses whole-grid ICM.

Launching and reviewing
----------------------

```bash
# GPU smoke: true 512 output, initial baseline equivalence, finite gradients.
sbatch experiments/bcss_wcsp/slurm/smoke.sbatch

# Inspect the entire submission graph without submitting jobs.
DRY_RUN=1 bash experiments/bcss_wcsp/slurm/submit_tonight.sh

# Submit training and dependent GPU prediction / CPU solve arrays.
bash experiments/bcss_wcsp/slurm/submit_tonight.sh
```

Training runs at most 3 GPUs concurrently. Each prediction array runs at
most 2 GPUs, one shard per task. Each CPU solve task uses 16 single-threaded
processes; 2 array tasks give 32 workers per treatment. There are 16 CPU
solve treatments in the default graph, so peak concurrency can be high.
Use SLURM dependencies or lower array limits if other workloads need space.
CPU workers each own their solver, avoiding Toulbar2 process-global state.

`EPOCHS`, `BATCH`, `DEV_LIMIT`, `PREDICT_SHARDS`, `WCSP_ENV`, `BCSS_ROOT`,
`WCSP_DATA`, and `WCSP_OUT` can be overridden before submission. Cache roots
and checkpoint names distinguish the training arms and full/center fields.
Use a fresh `WCSP_OUT` when replacing checkpoints or changing the dataset;
existing probability shards are reused without checking checkpoint hashes.
For `TRAIN_JOB=<array id>`, submission reuses that training array.

```bash
export WCSP_DATA="$PWD/data" WCSP_OUT="$PWD/data/tonight"
.venv/bin/python -m wcsp summarize \
  --config experiments/bcss_wcsp/configs/tonight/ce.yaml --run tonight_ce
.venv/bin/python -m wcsp compare \
  --config experiments/bcss_wcsp/configs/tonight/ce.yaml \
  --run tonight_pairwise_0.1 --against tonight_ce
```

Run names for additional experiments are `tonight_long_d8_p0.000001`,
`tonight_long_d8_p0.00001`, corresponding d32 names, and
`tonight_solver_{icm,toulbar2,hybrid}`. Use `pretrained.yaml` to summarize
those runs. Outputs live in `data/tonight/runs/<run>/dev/`; logs are in
`data/logs/`. `config.json` records every effective setting. Solver CSVs
record energies, changed-pixel fraction, runtime, and actual long-edge
count. ROI-paired bootstrap comparisons are already supported by `compare`.
