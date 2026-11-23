#!/bin/bash
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=bcss-full512-val-predict
#SBATCH --output=data/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TIATOOLBOX_HOME="$PWD/data/tiatoolbox" MPLCONFIGDIR="$PWD/data/matplotlib"
"${WCSP_ENV:-$PWD/.venv}/bin/python" -u experiments/bcss_wcsp/multiscale/fullres_validation.py predict \
  --root "${1:-$PWD/data/fullres_multiscale_val_20261115}" \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 8 --batch-size 4
