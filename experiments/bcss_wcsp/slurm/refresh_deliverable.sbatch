#!/bin/bash
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=bcss-report
#SBATCH --output=data/logs/%x_%j.out
# Submit with --dependency=afterany:<job ids> so failures/timeouts remain visible
# as incomplete results rather than preventing the report refresh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${WCSP_ENV:-$PWD/.venv}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/bcss_wcsp/update_deliverable.py --deliverable "$PWD/deliverables/deliverable_6.md" "$@"
