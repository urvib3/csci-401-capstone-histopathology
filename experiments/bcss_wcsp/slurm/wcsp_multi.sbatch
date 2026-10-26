#!/bin/bash
# Like wcsp.sbatch, but each array task runs one single-core `python -m wcsp`
# process per CPU, so a run is split into (array size x cpus) shards. Use this
# to get past the per-user running-job limit.
#
#   sbatch --array=0-7 experiments/bcss_wcsp/slurm/wcsp_multi.sbatch solve --config experiments/bcss_wcsp/configs/bcss_unet.yaml --split val --run NAME
#     -> 8 tasks x 16 processes = shards 0..127 of 128
#
# Do not pass --shard/--shards; they are set here.
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --job-name=wcsp-multi
#SBATCH --output=/scratch1/%u/wcsp_logs/%x_%A_%a.out

set -euo pipefail
module purge
module load gcc/13.3.0 python/3.11.9

source "${WCSP_ENV:-/scratch1/$USER/envs/wcsp}/bin/activate"
export WCSP_DATA="${WCSP_DATA:-/scratch1/$USER/data}"
export WCSP_OUT="${WCSP_OUT:-/scratch1/$USER/wcsp}"
export TMPDIR="/scratch1/$USER/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

per=$SLURM_CPUS_PER_TASK
total=$((SLURM_ARRAY_TASK_COUNT * per))
cd "$SLURM_SUBMIT_DIR"
for i in $(seq 0 $((per - 1))); do
    shard=$((SLURM_ARRAY_TASK_ID * per + i))
    python -m wcsp "$@" --shard "$shard/$total" > "${TMPDIR}/../wcsp_${SLURM_JOB_ID}_${shard}.log" 2>&1 &
done
fail=0
for job in $(jobs -p); do wait "$job" || fail=1; done
exit $fail
