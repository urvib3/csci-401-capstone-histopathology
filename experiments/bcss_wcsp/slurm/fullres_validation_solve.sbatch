#!/bin/bash
# Two fixed arms, each with 128 sample shards; 32 tasks x 8 CPU workers.
# Submit --array=0-31%4 to run at most 32 workers. Arms are interleaved.
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=bcss-full512-val-solve
#SBATCH --output=data/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root=${1:-$PWD/data/fullres_multiscale_val_20261115}
treatment=$((SLURM_ARRAY_TASK_ID % 2))
group=$((SLURM_ARRAY_TASK_ID / 2))
pids=()
for ((i=0; i<8; i++)); do
  shard=$((group * 8 + i))
  "${WCSP_ENV:-$PWD/.venv}/bin/python" -u experiments/bcss_wcsp/multiscale/fullres_validation.py solve \
    --root "$root" --treatment "$treatment" --shard "$shard" --shards 128 \
    > "data/logs/full512-val_${SLURM_ARRAY_JOB_ID}_${treatment}_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
