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
