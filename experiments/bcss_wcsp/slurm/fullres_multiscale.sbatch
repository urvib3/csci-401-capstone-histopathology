#!/bin/bash
# 18 treatments x 2 tasks x 8 single-threaded processes = 16 shards/treatment.
# At most four array tasks run together (32 CPU workers total).
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=bcss-full512-ms
#SBATCH --output=data/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
runner=experiments/bcss_wcsp/multiscale/fullres_sweep.py
root=${1:-$PWD/data/fullres_multiscale_20261115}
treatment=$((SLURM_ARRAY_TASK_ID / 2))
half=$((SLURM_ARRAY_TASK_ID % 2))
pids=()
for ((i=0; i<8; i++)); do
  shard=$((half * 8 + i))
  "${WCSP_ENV:-$PWD/.venv}/bin/python" -u "$runner" solve --root "$root" \
    --treatment "$treatment" --shard "$shard" --shards 16 \
    > "data/logs/full512-ms_${SLURM_ARRAY_JOB_ID}_${treatment}_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
