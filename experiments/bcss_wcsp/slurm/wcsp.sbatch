#!/bin/bash
# Run any `python -m wcsp` command as a CPU job; with --array, each task
# takes shard $SLURM_ARRAY_TASK_ID of --shards.
#
#   sbatch --array=0-19 experiments/bcss_wcsp/slurm/wcsp.sbatch predict --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split dev --shards 20
#   sbatch --array=0-19 experiments/bcss_wcsp/slurm/wcsp.sbatch solve   --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split dev --shards 20
#
# Override resources on the command line, e.g. sbatch --partition=debug --time=0:30:00 ...
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=wcsp
#SBATCH --output=/scratch1/%u/wcsp_logs/%x_%A_%a.out

set -euo pipefail
module purge
module load gcc/13.3.0 python/3.11.9

source "${WCSP_ENV:-/scratch1/$USER/envs/wcsp}/bin/activate"
export WCSP_DATA="${WCSP_DATA:-/scratch1/$USER/data}"
export WCSP_OUT="${WCSP_OUT:-/scratch1/$USER/wcsp}"
export TMPDIR="/scratch1/$USER/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TIATOOLBOX_HOME="${TIATOOLBOX_HOME:-/scratch1/$USER/tiatoolbox}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

cd "$SLURM_SUBMIT_DIR"
python -m wcsp "$@"
