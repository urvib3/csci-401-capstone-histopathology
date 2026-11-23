#!/bin/bash
# Class presence on Ryan's 185 frozen full-resolution dev inputs (rules/fullres_presence_dev.py).
#
#   sbatch --clusters=discovery experiments/bcss_wcsp/slurm/fullres_presence_dev.sbatch
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --job-name=fullres-presence
#SBATCH --output=/scratch1/%u/wcsp_logs/%x_%j.out

set -euo pipefail
module purge
module load gcc/13.3.0 python/3.11.9
source /scratch1/$USER/envs/wcsp/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python experiments/bcss_wcsp/rules/fullres_presence_dev.py --workers 16 "$@"
