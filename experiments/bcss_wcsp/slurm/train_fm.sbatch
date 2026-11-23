#!/bin/bash
# Fine-tune an open pathology foundation model + decoder on BCSS (experiments/bcss_wcsp/fm/train_fm.py).
# Open weights only (kaiko-ai/midnight default, owkin/phikon-v2); no HuggingFace login needed.
#
#   sbatch --clusters=discovery experiments/bcss_wcsp/slurm/train_fm.sbatch --encoder kaiko-ai/midnight --epochs 20 --unfreeze 4
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=fm-bcss
#SBATCH --output=/scratch1/%u/wcsp_logs/%x_%j.out

set -euo pipefail
module purge
module load gcc/13.3.0 python/3.11.9
source /scratch1/$USER/envs/vlm/bin/activate
export WCSP_DATA=/scratch1/$USER/data WCSP_OUT=/scratch1/$USER/wcsp
export TMPDIR=/scratch1/$USER/tmp HF_HOME=/scratch1/$USER/hf OMP_NUM_THREADS=8
mkdir -p "$TMPDIR" "$HF_HOME"

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
python experiments/bcss_wcsp/fm/train_fm.py "$@"
