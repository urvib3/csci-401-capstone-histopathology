#!/bin/bash
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=bcss-solve
#SBATCH --output=data/logs/%x_%A_%a.out
# One independent single-threaded solver process per CPU, sharded over tiles.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${WCSP_ENV:-$PWD/.venv}/bin/activate"
export WCSP_DATA="${WCSP_DATA:-$PWD/data}"
export WCSP_OUT="${WCSP_OUT:-$PWD/data/tonight}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
config=$1
run=$2
shift 2
per=$SLURM_CPUS_PER_TASK
total=$((SLURM_ARRAY_TASK_COUNT * per))
pids=()
for ((i=0; i<per; i++)); do
  shard=$((SLURM_ARRAY_TASK_ID * per + i))
  python -m wcsp solve --config "$config" --split dev --run "$run" \
    --limit "${DEV_LIMIT:-200}" --shard "$shard/$total" "$@" \
    > "data/logs/${SLURM_ARRAY_JOB_ID}_${shard}.log" 2>&1 &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit "$fail"
