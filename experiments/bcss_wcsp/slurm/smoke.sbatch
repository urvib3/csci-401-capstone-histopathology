#!/bin/bash
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=bcss-smoke
#SBATCH --output=data/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/.venv/bin/activate"
export TIATOOLBOX_HOME="$PWD/data/tiatoolbox"
export HF_HOME="$PWD/data/huggingface"
export MPLCONFIGDIR="$PWD/data/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u experiments/bcss_wcsp/smoke_training.py
