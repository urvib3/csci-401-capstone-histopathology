#!/bin/bash
# Run from any directory. DRY_RUN=1 prints commands without submitting jobs.
# --test-only asks the Discovery scheduler to validate resources without submitting.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
if (($# > 1)) || [[ ${1:-} != '' && ${1:-} != --test-only ]]; then
  echo "Usage: [DRY_RUN=1] bash $0 [--test-only]" >&2
  exit 2
fi
export WCSP_ENV=${WCSP_ENV:-$PWD/.venv}
root=$PWD/data/fullres_expanded_20261122_v2
slurm=experiments/bcss_wcsp/slurm
mkdir -p data/logs
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'sbatch --parsable --chdir=%q %q %q\n' "$PWD" "$slurm/fullres_expanded.sbatch" "$root"
  printf 'sbatch --parsable --chdir=%q --dependency=afterany:<sweep_job_id> %q %q\n' \
    "$PWD" "$slurm/fullres_expanded_report.sbatch" "$root"
  exit 0
fi
# Verify source checksums before allocating the full array. The runner checks
# each frozen input checksum again before solving its tile.
"$WCSP_ENV/bin/python" experiments/bcss_wcsp/multiscale/fullres_expanded.py prepare --root "$root"
if [[ ${1:-} == --test-only ]]; then
  sbatch --test-only --chdir="$PWD" "$slurm/fullres_expanded.sbatch" "$root"
  sbatch --test-only --chdir="$PWD" "$slurm/fullres_expanded_report.sbatch" "$root"
  exit 0
fi
sweep=$(sbatch --parsable --chdir="$PWD" "$slurm/fullres_expanded.sbatch" "$root")
# --clusters adds ';discovery' to the parsable response; dependencies need only ID.
sweep_id=${sweep%%;*}
printf 'Discovery sweep: %s\n' "$sweep_id"
# Report partial progress even if an array task fails or hits its wall limit.
sbatch --parsable --chdir="$PWD" --dependency="afterany:$sweep_id" \
  "$slurm/fullres_expanded_report.sbatch" "$root"
