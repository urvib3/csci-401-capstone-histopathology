#!/bin/bash
#SBATCH --clusters=discovery
#SBATCH --account=dilkina_1463
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=bcss-corrections-report
#SBATCH --output=data/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ ! -f data/fullres_corrections_20261122/protocol.json ]]; then
  echo 'Preparation did not freeze a protocol; inspect preparation log.'
  exit 1
fi
"${WCSP_ENV:-$PWD/.venv}/bin/python" -u experiments/bcss_wcsp/multiscale/fullres_corrections.py report
