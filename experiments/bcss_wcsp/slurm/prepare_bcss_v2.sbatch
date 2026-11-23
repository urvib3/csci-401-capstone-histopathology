#!/bin/bash
# Download the BCSS ROIs at 0.25 MPP from the public HistomicsTK server and tile them
# at 512 px (preparation v2: 7,768 fit / 1,012 dev / 4,028 val), as Ryan did.
#
#   sbatch --clusters=discovery experiments/bcss_wcsp/slurm/prepare_bcss_v2.sbatch
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=bcss-v2
#SBATCH --output=/scratch1/%u/wcsp_logs/%x_%j.out

set -euo pipefail
module purge
module load gcc/13.3.0 python/3.11.9
source /scratch1/$USER/envs/wcsp/bin/activate
export TMPDIR=/scratch1/$USER/tmp OMP_NUM_THREADS=1
mkdir -p "$TMPDIR"

cd "$SLURM_SUBMIT_DIR"
python scripts/prepare_bcss_upstream.py --out /scratch1/$USER/data/bcss_v2 --workers 4
du -sh /scratch1/$USER/data/bcss_v2
