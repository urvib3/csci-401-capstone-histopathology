# Full-resolution U-Net multiscale follow-up

The original multiscale [mu sweep](TODO.md) used the pretrained half-resolution
U-Net and the original central-256 tiling. It selected mu=0.5, three levels.
This follow-up repeats mu={0, 0.05, 0.1, 0.2, 0.5, 1} with the local trained
full-resolution checkpoint. No exponential decay parameter is introduced.

The existing `HierarchySolver` is used unchanged. Its energy includes pixel
unaries and neighbors, block unaries and neighbors, and
`mu * child_pixel_area * [child_label != parent_label]` for each parent-child
link. Three block levels have sides 4, 8 and 16 pixels. A conditioned pixel
variable pays mu for disagreeing with its parent. A coarser child pays mu
times its area; mu is a consistency strength rather than a probability decay.

All 18 treatments reuse the same cached 5x512x512 probabilities: 185 scored
tiles from the earlier evenly spaced 200-tile dev selection, preparation v2.
The trained checkpoint is `data/outputs/tonight/full512/best.pt`; its SHA256
is `9cceca988b77f0b08b276a7dac0d8ee83e9574e922fe93bfdddb8d6759fd8e17`.
The byte-identical repository artifact is
[`models/bcss_fullres_unet/best.pt`](../../../models/bcss_fullres_unet/best.pt),
stored with Git LFS. [Model documentation](../../../models/bcss_fullres_unet/README.md)
explains download, checksum verification, loading and restoring the historical
local checkpoint path. The submitted jobs retain their frozen original paths.
The argmax baseline is mIoU 0.6524144074620204. These development results are
not directly comparable to the original multiscale sweep's different tiling
and scored field.

Pixel and block neighbor weights remain lambda=8; alpha=1; temperature=1;
the data-derived table and other multiscale settings match the documented
multiscale experiment. The earlier full-resolution ICM result (0.654473)
used a handwritten table and whole-grid ICM, so the new mu=0 row for each
solver is the matched control. It also controls for extra hierarchy rounds.
Pixel solvers retain the documented 100-pixel corners (a 312-pixel cross at
512 resolution), four block rounds and 50 ICM sweeps per call. Methods:
ICM, Toulbar2, and ICM then Toulbar2 within each conditioned block.
Toulbar2 uses the earlier 2-second search cap per block; build time is extra.
Block hierarchy levels always use ICM. Four outer hierarchy rounds are
allowed, with early stopping. Finite search caps do not establish optimality.

Preparation freezes inputs and hashes source/config/checkpoint/table/cache
files. Each completed tile saves pixel and block labels, confusion matrices,
runtime and the hierarchy's level-update energy trace. Outputs resume at
tile boundaries. The ordinary pixel energy and augmented hierarchy energy
are stored separately; energies for different mu must not be ranked together.
Reports withhold aggregate metrics for incomplete populations and include
ROI-paired bootstrap comparisons with each method's mu=0 control when both
are complete. No validation evaluation is launched or used for selection.

From the repository root:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python experiments/bcss_wcsp/multiscale/fullres_sweep.py prepare
# Smoke one full-resolution dev tile at mu=0.5 for each method.
for treatment in 4 10 16; do
  .venv/bin/python experiments/bcss_wcsp/multiscale/fullres_sweep.py solve \
    --treatment "$treatment" --smoke
done
sbatch --array=0-35%4 experiments/bcss_wcsp/slurm/fullres_multiscale.sbatch
.venv/bin/python experiments/bcss_wcsp/multiscale/fullres_sweep.py report
```

Outputs and the immutable `protocol.json` live in
`data/fullres_multiscale_20261115/`. A new output root is required after
changing any frozen source, input, checkpoint or treatment configuration.
The sweep runs at most 32 CPU workers across four 8-CPU tasks. Automatic
reporting should depend on `afterany:SWEEP_JOB_ID` to expose failed runs.

## Submission on November 15

Sweep array **7029728** (`0-35%4`); report job **7029765**, dependent on
`afterany:7029728`. Both were accepted by SLURM; the array was initially
pending for priority. IDs and resource settings are also in `jobs.json`
under the output root. The automatic report writes `REPORT.md`,
`summary.csv` and `summary.json` there.

Seven focused tests passed, and all three methods completed a real 512px
smoke tile at mu=0.5 with nonincreasing augmented energy and valid pixel
and block outputs. On this tile, capped Toulbar2 alone changed no pixel
labels; ICM and the hybrid produced identical pixel labels. These are
implementation checks, not aggregate accuracy results.

A fresh CPU inference using the recorded checkpoint agreed with the cached
GPU argmax on 99.913% of this tile's pixels. Mean absolute probability
difference was 0.000583; the maximum was 0.030464, so the outputs are not
numerically identical. The sweep uses the frozen historical GPU probability
arrays consistently across every arm, rather than mixing regenerated CPU
predictions into the comparison. The diagnostic is saved locally as
`checkpoint_cache_cpu_check.json`.
