#!/bin/bash
# 17 treatments x 4 tasks x 8 workers = 32 shards/treatment, at most 6 tiles/worker.
#SBATCH --clusters=discovery
#SBATCH --account=dilkina_1463
#SBATCH --partition=main
#SBATCH --array=0-67%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=bcss-expanded
#SBATCH --output=data/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root=${1:-$PWD/data/fullres_expanded_20261122_v2}
python=${WCSP_ENV:-$PWD/.venv}/bin/python
if [[ ! -x "$python" || ! -f "$root/protocol.json" ]]; then
  echo "Missing Python environment or frozen protocol: $python, $root/protocol.json" >&2
  exit 1
fi
if [[ ! ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ ]] || ((SLURM_ARRAY_TASK_ID > 67)); then
  echo "Expected an array task ID in 0-67" >&2
  exit 1
fi
treatment=$((SLURM_ARRAY_TASK_ID / 4))
group=$((SLURM_ARRAY_TASK_ID % 4))
pids=()
for ((i=0; i<8; i++)); do
  shard=$((group * 8 + i))
  "$python" -u experiments/bcss_wcsp/multiscale/fullres_expanded.py solve \
    --root "$root" --treatment "$treatment" --shard "$shard" --shards 32 \
    > "data/logs/bcss-expanded_${SLURM_ARRAY_JOB_ID}_${treatment}_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
