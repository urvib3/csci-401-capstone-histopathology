#!/bin/bash
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=bcss-predict
#SBATCH --output=data/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${WCSP_ENV:-$PWD/.venv}/bin/activate"
export WCSP_DATA="${WCSP_DATA:-$PWD/data}"
export WCSP_OUT="${WCSP_OUT:-$PWD/data/tonight}"
export TIATOOLBOX_HOME="${TIATOOLBOX_HOME:-$PWD/data/tiatoolbox}"
export HF_HOME="${HF_HOME:-$PWD/data/huggingface}"
export MPLCONFIGDIR="$PWD/data/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Eight shards keep the full-size dev probability files modest.
python -m wcsp predict --config "$1" --split dev --shards "${PREDICT_SHARDS:-8}" \
  --limit "${DEV_LIMIT:-200}" --batch-size 4 --set classifier.device=cuda
