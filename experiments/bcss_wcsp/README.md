# BCSS WCSP experiments

This directory contains the configurations, SLURM scripts, and figure-generation code for pixel-level weighted constraint satisfaction problem (WCSP) experiments on the BCSS histopathology dataset.

The experiments use a U-Net to produce tissue-class probabilities, then apply pairwise costs to refine the pixel labels. The WCSP implementation and command-line interface are documented in [`wcsp/README.md`](../../wcsp/README.md). The overnight experiment plan and its operational details are in [`TONIGHT.md`](TONIGHT.md).

## Labels

The five BCSS classes are tumor, stroma, inflammatory, necrosis, and other. The WCSP code uses this internal order:

1. Neoplastic
2. Inflammatory
3. Connective
4. Dead
5. Healthy

BCSS labels are remapped to this order when loaded. BCSS “other” is assigned to the healthy slot in the pairwise cost table.

## Data preparation and splits

The newer experiment workflow prepares BCSS data from the upstream dataset using `scripts/prepare_bcss_upstream.py`. It downloads images and masks, creates nonoverlapping 512-pixel tiles, and records preparation details in a manifest.

The planned split is by ROI:

- Hospitals OL, LL, E2, EW, GM, and S3 are held out for validation.
- Every eighth sorted training ROI is assigned to development.
- The remaining training ROIs are used for fitting.
- Label codes 0 and 7 are ignored.
- Validation labels are not used to select models or parameters.

This preparation creates a fresh tiling of upstream BCSS data. It may not match the repository’s earlier benchmark tiles. Record the preparation version and manifest when comparing experiments.

Run the preparation script from the repository root:

```bash
.venv/bin/python scripts/prepare_bcss_upstream.py --workers 4
```
# BCSS WCSP Experiments

Before launching training, verify that preparation completed and that `data/bcss/manifest.json` exists. The overnight run plan expects preparation version 2 and all source ROIs to be present. See [`TONIGHT.md`](TONIGHT.md) for the full checks and storage requirements.

## Image and prediction fields

Input tiles are **512 × 512** pixels. The standard pretrained TIAToolbox U-Net produces five class-probability maps at **256 × 256**. The maps are bilinearly upsampled to 512 × 512. The standard WCSP configuration evaluates and solves only the **central 256 × 256** area.

The central crop contains one variable per pixel. Each variable can take one of the five BCSS labels. The U-Net baseline assigns each pixel the class with the highest probability.

Alternative configurations include:

- **Native-resolution inference:** `tia_unet_native` uses shift-and-stitch inference to form per-pixel predictions from shifted inputs without retraining the U-Net.
- **Full-resolution inference:** `tia_unet_fullres` uses a full-resolution head to produce predictions over the full 512 × 512 tile.
- **Full-tile evaluation:** selected configurations score the full field rather than the central crop. Check the configuration's `dataset.full_tile` setting to see which field it uses.

## WCSP formulation

The WCSP objective combines unary costs with pairwise costs:

```text
total cost = unary cost + λ × pairwise cost
```

- **Unary cost:** `−log(p)`, where `p` is the U-Net probability for a candidate class.
- **Pairwise cost:** a penalty from a class-pair table for two neighboring pixels, scaled by their distance.
- **λ:** controls the strength of pairwise costs relative to the unary costs.
- **η:** controls how pairwise costs vary with distance.

The local pairwise setup uses five neighborhood types. Configurations select the table, neighborhood types, normalization, and solver. The available tables are in [`configs/tables/`](configs/tables/).

The package supports **ICM** and **Toulbar2** solvers, as well as configured solver combinations. The standard experiment uses a block-based solver configuration; some newer configurations use whole-grid ICM. See each YAML file for its solver settings.

## Experiment configurations

### Standard WCSP configurations

The main configurations are:

- [`configs/bcss_unet.yaml`](configs/bcss_unet.yaml): standard pretrained U-Net probabilities and local WCSP.
- [`configs/bcss_unet_native.yaml`](configs/bcss_unet_native.yaml): native-resolution shift-and-stitch inference with WCSP.
- [`configs/bcss_unet_fullres.yaml`](configs/bcss_unet_fullres.yaml): full-resolution U-Net configuration.

Development sweeps are defined in [`configs/sweeps/`](configs/sweeps/). Pairwise cost tables are in [`configs/tables/`](configs/tables/).

### Training configurations

The newer experiment plan defines training arms under [`configs/tonight/`](configs/tonight/):

| Configuration | Model and field | Training loss |
| --- | --- | --- |
| `ce.yaml` | Standard half-resolution model; central field | Cross-entropy |
| `full512.yaml` | Full-resolution head; full tile | Cross-entropy |
| `pairwise_0.01.yaml` | Standard model; central field | Cross-entropy plus weighted expected pairwise cost |
| `pairwise_0.1.yaml` | Standard model; central field | Cross-entropy plus weighted expected pairwise cost |
| `pairwise_1.yaml` | Standard model; central field | Cross-entropy plus weighted expected pairwise cost |

The training-loss weight is separate from the inference WCSP `λ`. The pairwise training loss uses the handwritten table and masks pairs that touch ignored pixels. It does not run a WCSP solver during backpropagation.

Additional configurations include `pretrained.yaml`, `ce_full512.yaml`, and `pretrained_full512.yaml` for evaluating pretrained or trained models over the selected fields.

### Sampled long-distance pairs and solver comparisons

The overnight plan also defines experiments that sample distant pixel pairs and compare solver choices. The distant-pair settings are controlled through configuration overrides for distance and sampling probability. See [`TONIGHT.md`](TONIGHT.md) for the configurations and launch commands.

The plan includes comparisons among ICM, Toulbar2, and a chained ICM-then-Toulbar2 solver. When reviewing runs, use the recorded effective configuration and compare runtime and objective values alongside segmentation metrics.

## Running experiments

Run these commands from the repository root. Set the data and output locations before running:

```bash
export WCSP_DATA="$PWD/data"
export WCSP_OUT="$PWD/data/outputs"
```

Cache predictions, solve the WCSP, and summarize a run:

```bash
python -m wcsp predict \
  --config experiments/bcss_wcsp/configs/bcss_unet.yaml \
  --split dev

python -m wcsp solve \
  --config experiments/bcss_wcsp/configs/bcss_unet.yaml \
  --split dev \
  --run my_run \
  --set problem.lam=8

python -m wcsp summarize \
  --config experiments/bcss_wcsp/configs/bcss_unet.yaml \
  --split dev \
  --run my_run
```

For the overnight experiment graph, inspect the job commands without submitting them:

```bash
DRY_RUN=1 bash experiments/bcss_wcsp/slurm/submit_tonight.sh
```

To submit the configured jobs:

```bash
bash experiments/bcss_wcsp/slurm/submit_tonight.sh
```

The submission script expects the prepared dataset, a working Python environment, and the TIAToolbox model weights. See [`TONIGHT.md`](TONIGHT.md) for environment variables, job dependencies, concurrency settings, output directories, and run names.

Use a fresh output directory when changing the dataset or model checkpoints. Cached probability shards may be reused without checking whether they match the current checkpoint.

## Testing

The instrumented [expanded full-resolution runner](multiscale/fullres_expanded.py)
reuses the frozen 185-tile dev inputs for 15 fully converged combinations of
mu={0, 0.5, 1, 2, 4} and lambda={4, 8, 16}, tying pixel and block neighbor
lambda. Two additional mu=0/1 runs retain the original inner limits and save
4/8/12-outer-round checkpoints before continuing to convergence. The new
[solver adapter](multiscale/convergence_solver.py) records every ICM sweep,
label changes and stopping reasons, and audits single-variable stationarity
at the pixel and all coarse levels. This establishes a local minimum within
1e-9 tolerance, with no claim of a global optimum. Historical solvers and
the completed validation benchmark remain unchanged.

The frozen expanded protocol and outputs live under
`data/fullres_expanded_20261122_v2/`; unfinished tiles retain live iteration
logs and are excluded from complete-population accuracy reports. Run
`python experiments/bcss_wcsp/multiscale/fullres_expanded.py report` to refresh
the local report. The batch script is
[`slurm/fullres_expanded.sbatch`](slurm/fullres_expanded.sbatch), targeting
Discovery's `main` CPU partition with account `dilkina_1463`. Its default
`--array=0-67%4` gives 32 shards per treatment: 68 array tasks, each with
8 CPUs and 64 GiB RAM, with at most 4 tasks (32 CPU workers) running together.
Each task has a 12-hour allocation; that wall limit can interrupt an
unconverged tile, which remains explicitly incomplete. Completed tiles are
reused on resubmission; an interrupted tile restarts from its initial labels.

The new experiment inventory is:

| Treatments | Parameters | Question |
|---|---|---|
| 15 fully converged runs | lambda={4, 8, 16} crossed with mu={0, 0.5, 1, 2, 4} | How do smoothing and cross-scale coupling affect accuracy once all iteration caps are removed? |
| 2 outer convergence curves | lambda=8, mu={0, 1}; retain 4 pixel block rounds and 50 ICM sweeps per inner/coarse call | How do the historical 4-outer-round results compare with 8/12 rounds and convergence? |

Every treatment uses the same 185 frozen 512x512 development tiles, cached
U-Net probabilities, and three coarse levels. This totals 3,145 tile solves;
the 4/8/12-round curve checkpoints are saved during those solves, rather than
separate jobs. A run converging before a checkpoint reuses its final labels
for that checkpoint. mu=0 is the matched control for each lambda or curve.
Outputs include labels at every level, live iteration traces, energy,
changed-label counts, convergence audits, runtime, mIoU/Dice and paired
ROI-bootstrap comparisons against mu=0. Aggregate accuracy is withheld until
all 185 tiles of a treatment/checkpoint are complete. These runs require no
GPU, retraining, new inference, or validation sweep.

Preview commands or check scheduling without submitting:

```bash
DRY_RUN=1 bash experiments/bcss_wcsp/slurm/submit_fullres_expanded.sh
bash experiments/bcss_wcsp/slurm/submit_fullres_expanded.sh --test-only
```

When ready, submit the sweep and its dependent report:

```bash
bash experiments/bcss_wcsp/slurm/submit_fullres_expanded.sh
```

The [submission helper](slurm/submit_fullres_expanded.sh) verifies the frozen
protocol before submission. The [report job](slurm/fullres_expanded_report.sbatch)
runs after the entire array ends, including failures, and updates
`data/fullres_expanded_20261122_v2/REPORT.md` and `summary.json`. It requests
1 CPU, 8 GiB RAM, and 1 hour. The older `fullres_expanded_20261122/` root is
superseded; the completed historical development and validation sweeps are
separate from these new jobs.

### Full-512 correction follow-ups

The [correction runner](multiscale/fullres_corrections.py) and
[method implementations](multiscale/correction_methods.py) add the following
40 development treatments. Every treatment uses all 185 frozen full-512
inputs (7,400 tile solves total), the same checkpoint and lambda=8.

| Family | Configurations | Count |
|---|---|---:|
| Temperature | T={1,1.5,2,3,4}, each with mu={0,1} | 10 |
| Block fidelity | block_alpha={0,0.25,0.5}, mu=1; alpha=1 control is shared with T=1 | 3 |
| Tile presence | gamma={40,120,400,1200,4000,12000}, mu={0,1}; zero-strength controls are shared | 12 |
| Coordinated region proposals | Connected components or image-derived SLIC, mu={0,1} | 4 |
| Image-aware boundaries | HED contrast_beta={1,4}, mu={0,1} | 4 |
| Independent regional evidence | Appearance unary weight={0.1,0.3,1}, mu={0,1} | 6 |
| Appearance control | Independent regional classifier without U-Net or WCSP smoothing | 1 |

The [preparation job](slurm/corrections_prepare.sbatch) fits a smoothed
class-presence prior on all eligible **full-field fit masks**, requiring at
least 1,000 labeled pixels and defining presence as >=1% of labeled area.
It verifies fit/dev ID and ROI separation. For independent appearance evidence,
288 evenly spaced fit tiles are split into 256 fitting and 32 calibration
tiles, with every ninth tile reserved for calibration. A diagonal Gaussian
classifier uses HED stain channels, their sigma-4 local means and local
standard deviations. It fits on up to 128 pixels per class per fitting tile,
uses uniform class priors and regularized variances, and selects its temperature
only on the reserved fit tiles. This is a deliberately simple complementary
appearance model, not another U-Net training run.

Development RGB images are matched by ID and their current annotations must
exactly equal the frozen targets. SLIC uses 256 requested segments,
compactness=10, sigma=1 and connectivity enforcement. Appearance probabilities
are averaged within each segment to form regional unaries. RGB, SLIC and
regional unaries are cached with checksums under
`data/fullres_corrections_20261122/assets/`. Development targets only verify
alignment and score outputs; they do not construct image regions or appearance
evidence. In legacy WCSP metric keys, `healthy` denotes BCSS **other**, not
healthy epithelial tissue.

The presence method retains the existing two-stage formulation: select an
allowed class set using pixel evidence plus a fit-only co-occurrence prior,
then add a 1,000 unary exclusion penalty to other classes before solving.
Every result checks that no excluded label survives. The new gamma range
spans weak to strong priors at 512px and includes matched zero-prior controls.

Region methods first reach the ordinary hierarchy's single-variable local
minimum, then test up to two passes over the 32 largest eligible regions
(minimum 64 pixels). Each candidate assigns a region one tissue class,
either alone or jointly with coarse blocks at least half covered by it.
Acceptance uses the exact full hierarchy objective; resumed pixel/coarse
descent cannot raise that objective. Final stationarity is audited at all
levels. The finite proposal budget does not establish region or global
optimality. SLIC and connected-component runs test the optimizer on the same
energy as their matched temperature-one controls.

All other iterative solvers are uncapped until audited stationarity. Live
JSONL traces and atomic per-tile results support restarting incomplete tasks.
The report includes pooled and per-class metrics, ROI-paired intervals versus
the U-Net and matched mu control, fixes/breaks/wrong-to-wrong edits, and
boundary versus >10px-interior edits. Solver runtime excludes cached feature
preparation; preparation is logged separately. Accuracy is withheld for
incomplete 185-tile populations. Temperature/appearance changes also affect
the hierarchy's reconstructed coarse probabilities; block_alpha isolates
coarse fidelity without changing the pixel unary.

Preview, validate or submit from any directory:

```bash
DRY_RUN=1 bash experiments/bcss_wcsp/slurm/submit_corrections.sh
bash experiments/bcss_wcsp/slurm/submit_corrections.sh --test-only
bash experiments/bcss_wcsp/slurm/submit_corrections.sh
```

The [submission helper](slurm/submit_corrections.sh) schedules preparation and
four real full-512 smoke checks, then releases the 160-task correction array
only if those checks pass. Each task requests 8 CPUs, 64 GiB and 12 hours;
at most four tasks run together (32 new workers, in addition to the original
expanded sweep). A dependent report runs after preparation and the array end,
including failures. Submission IDs are appended to
`data/fullres_corrections_20261122/submissions.jsonl`. Frozen protocol, fitting
provenance, `REPORT.md` and `summary.json` share that output root. These
experiments do not evaluate validation data or change the original sweep.

The [shared development inputs](../../data/fullres_multiscale_20261115/README.md)
provide the 185 cached full-resolution probability/ground-truth tiles through
Git LFS, plus the original `protocol.json`, so teammates can compare heuristics
without access to Ryan's private cluster paths.

The [full-resolution validation protocol](multiscale/FULLRES_VALIDATION.md)
evaluates dev-selected ICM mu=1 against mu=0 and the raw U-Net on all 4,028
validation tiles, with full 512px outputs and frozen settings.

The [full-resolution multiscale follow-up](multiscale/FULLRES.md) repeats the
documented mu sweep with the local trained 512px U-Net, paired ICM/Toulbar2
controls, frozen probability inputs, and resumable per-tile outputs.

Run the WCSP package tests from the repository root:

```bash
python -m pytest tests/wcsp
```

## Repository map

- [`configs/`](configs/): experiment configurations
- [`configs/sweeps/`](configs/sweeps/): development sweep definitions
- [`configs/tables/`](configs/tables/): pairwise cost tables
- [`configs/tonight/`](configs/tonight/): training and evaluation configurations for the overnight experiment plan
- [`slurm/`](slurm/): batch scripts and experiment submission helpers
- [`figures/`](figures/): pipeline, WCSP, and example figures
- [`make_figures.py`](make_figures.py): figure-generation script
- [`smoke_training.py`](smoke_training.py): training smoke-check helper
- [`TONIGHT.md`](TONIGHT.md): detailed overnight run plan
- [`../../wcsp/README.md`](../../wcsp/README.md): WCSP package and CLI documentation

## Figures

Figures in [`figures/`](figures/) include the pipeline overview, WCSP diagram, example outputs, and a stroma-region visualization. To generate figures for a tile, run this command from the repository root:

```bash
python experiments/bcss_wcsp/make_figures.py --id <tile_id>
```
