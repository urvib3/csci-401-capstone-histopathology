#!/bin/bash
# Loss ablation on Ryan's recipe (original 256-centre architecture, 10 epochs, batch 4,
# seed 0, lr 1e-4), changing only the loss. Array task = arm:
#   0  A  pixel loss (cross-entropy)
#   1  B  + region consistency (ground-truth connected regions)
#   2  C  + tile-level presence (BCE on the true class set)
#
#   sbatch --clusters=discovery --array=0-2 experiments/bcss_wcsp/slurm/train_loss_arms.sbatch
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=bcss-loss-arms
#SBATCH --output=/scratch1/%u/wcsp_logs/%x_%A_%a.out

set -euo pipefail
module purge
module load gcc/13.3.0 python/3.11.9
source /scratch1/$USER/envs/train/bin/activate
export TMPDIR=/scratch1/$USER/tmp OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"

arms=("A_ce" "B_ce_cluster" "C_ce_presence")
extra=("" "--cluster-weight 1" "--presence-weight 1")
arm=${arms[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
# shellcheck disable=SC2086
python benchmarks/bcss/scripts/train_experiments.py --architecture baseline --field center \
    --epochs 10 --batch 4 --seed 0 --root /scratch1/$USER/data/bcss_v2 \
    --out /scratch1/$USER/wcsp/train_arms/$arm ${extra[$SLURM_ARRAY_TASK_ID]}
