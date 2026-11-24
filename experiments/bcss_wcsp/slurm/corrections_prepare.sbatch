#!/bin/bash
#SBATCH --clusters=discovery
#SBATCH --account=dilkina_1463
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=bcss-correction-prepare
#SBATCH --output=data/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=${WCSP_ENV:-$PWD/.venv}/bin/python
runner=experiments/bcss_wcsp/multiscale/fullres_corrections.py
"$python" -u "$runner" prepare
"$python" -u "$runner" smoke
"$python" -u "$runner" report
