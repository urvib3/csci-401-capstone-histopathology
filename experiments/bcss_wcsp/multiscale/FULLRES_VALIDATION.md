# Full-resolution BCSS validation

This is a fixed evaluation of the best completed **ICM** development setting,
not another mu sweep. The entire six-value ICM grid completed on the paired
185-tile dev selection. It selected **mu=1**, three levels, dev mIoU
0.656822 versus 0.654690 for its mu=0 control. The hybrid development sweep
was still running when this separate ICM validation protocol was defined.
Its future results will not change this validation selection.

The two validation arms are ICM with mu=0 and ICM with mu=1, with the raw
U-Net predictions scored as their shared baseline. They use the same trained
full-resolution checkpoint, data-derived compatibility table, lambda=8,
alpha=1 and temperature=1 as the development sweep. Block sides remain
4, 8 and 16 pixels; the pixel solver keeps 100-pixel corner blocks, four
block rounds and up to 50 ICM sweeps per call. The hierarchy allows four
outer rounds and stops early on an energy plateau. The heuristic is unchanged.

Every preparation-v2 **validation tile (4,028 tiles from 45 held-out ROIs)**
is processed at full **512x512** resolution. There is no tile limit,
subsampling or central crop. Codes 0 and 7 remain ignored. A tile with no
labeled pixels is recorded, excluded from aggregate metrics, and requires
no inference or solve; the report states total, scored and wholly ignored
tile counts separately.

`fullres_validation.py freeze` writes an immutable `selection.json` before
validation inference. It records the completed dev selection, dev artifact
hashes, validation IDs and source/model/configuration hashes. Model selection
does not inspect validation scores. No training or calibration occurs here.
The checkpoint remains `data/outputs/tonight/full512/best.pt`, SHA256
`9cceca988b77f0b08b276a7dac0d8ee83e9574e922fe93bfdddb8d6759fd8e17`.
Existing development sources and output files are preserved.

## Execution

Run from the repository root, using `.venv`:

1. `python experiments/bcss_wcsp/multiscale/fullres_validation.py freeze`
2. Submit `slurm/fullres_validation_predict.sbatch` with `--array=0-7%2`.
   Eight GPU shards stream batches of four into atomic per-tile float16
   probability/target files. At most two GPUs run at once.
3. After **all** prediction tasks succeed, run
   `python experiments/bcss_wcsp/multiscale/fullres_validation.py prepare`.
   This requires metadata for all 4,028 IDs and verifies input checksums.
4. After preparation succeeds, submit `slurm/fullres_validation_solve.sbatch`
   with `--array=0-31%4`: 128 shards per arm, eight single-threaded CPU workers
   per task, at most 32 workers overall. Array tasks interleave the two arms.
5. After the solve array ends (including failed/time-limited tasks), run
   `python experiments/bcss_wcsp/multiscale/fullres_validation.py report`.

The script paths in steps 2 and 4 are under `experiments/bcss_wcsp/`.
Each solve task has an eight-hour allocation. Completed tiles are saved
individually and reused if a failed task is resubmitted. Source/model/input
hash mismatches stop the run. The shared GPU predictions are used by both
arms; CPU inference is not mixed into the comparison.

Outputs live in **`data/fullres_multiscale_val_20261115/`**:

- `selection.json`: frozen dev-only selection and provenance.
- `inputs/`: per-tile metadata and probabilities/targets; wholly ignored tiles
  have metadata only.
- `protocol.json`: complete scored population, ignored IDs and raw baseline.
- `runs/<arm>/val/samples/`: final pixel/block labels, confusion matrices,
  runtime and level-update energy traces. Pixel and augmented hierarchy
  energies are stored separately.
- `summary.json`, `REPORT.md`: complete-population results. Incomplete arms
  receive counts/status only, with no partial aggregate accuracy. JSON includes
  per-class metrics and ROI-paired bootstrap comparisons against the raw
  U-Net and the matching mu=0 control.
- `jobs.json`: submitted job IDs and dependencies.

Runtime is observed under concurrent cluster load, not a controlled hardware
benchmark. The validation report evaluates the frozen ICM choice; it does
not choose a new mu or solver using validation performance.

## Submitted jobs

Submitted November 15 with the frozen mu=1 choice:

- **7033605**: GPU prediction array (`0-7%2`).
- **7033613**: complete-population preparation (`afterok:7033605`).
- **7033615**: fixed-arm CPU solve array (`0-31%4`, `afterok:7033613`).
- **7033616**: final report (`afterany:7033615`).

The first two prediction tasks started running after submission. Job IDs
and dependencies are also saved in the local output root's `jobs.json`.
