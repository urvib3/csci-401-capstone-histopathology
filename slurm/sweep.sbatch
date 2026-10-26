#!/bin/bash
# One `python -m wcsp solve` per line of a sweep file; array task i runs line i+1.
# Each line is "<run name> <extra solve arguments>".
#
#   n=$(wc -l < configs/sweeps/bcss_dev.txt)
#   sbatch --array=0-$((n-1)) slurm/sweep.sbatch configs/bcss_unet.yaml configs/sweeps/bcss_dev.txt --split dev --limit 200
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=2:00:00
#SBATCH --job-name=wcsp-sweep
#SBATCH --output=/scratch1/%u/wcsp_logs/%x_%A_%a.out

set -euo pipefail
config=$1 sweep=$2
shift 2
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$sweep")
read -r run extra <<< "$line"

module purge
module load gcc/13.3.0 python/3.11.9
source "${WCSP_ENV:-/scratch1/$USER/envs/wcsp}/bin/activate"
export WCSP_DATA="${WCSP_DATA:-/scratch1/$USER/data}"
export WCSP_OUT="${WCSP_OUT:-/scratch1/$USER/wcsp}"
export OMP_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC2086
python -m wcsp solve --config "$config" --run "$run" --shard 0/1 $extra "$@"
python -m wcsp summarize --config "$config" --run "$run" "$@" > /dev/null
