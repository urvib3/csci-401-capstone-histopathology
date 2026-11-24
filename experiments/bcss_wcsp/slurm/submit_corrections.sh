#!/bin/bash
# DRY_RUN=1 previews commands; --test-only validates resources without submission.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
if (($# > 1)) || [[ ${1:-} != '' && ${1:-} != --test-only ]]; then
  echo "Usage: [DRY_RUN=1] bash $0 [--test-only]" >&2
  exit 2
fi
export WCSP_ENV=${WCSP_ENV:-$PWD/.venv}
mkdir -p data/logs
slurm=experiments/bcss_wcsp/slurm
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%s\n' "sbatch $slurm/corrections_prepare.sbatch" \
    "sbatch --dependency=afterok:PREP_ID --kill-on-invalid-dep=yes $slurm/corrections_sweep.sbatch" \
    "sbatch --dependency=afterany:PREP_ID:SWEEP_ID $slurm/corrections_report.sbatch"
  exit 0
fi
if [[ ${1:-} == --test-only ]]; then
  for script in corrections_prepare corrections_sweep corrections_report; do
    sbatch --test-only --chdir="$PWD" "$slurm/$script.sbatch"
  done
  exit 0
fi
prep=$(sbatch --parsable --chdir="$PWD" "$slurm/corrections_prepare.sbatch")
prep=${prep%%;*}
printf 'Preparation and full-512 smoke checks: %s\n' "$prep"
sweep=$(sbatch --parsable --chdir="$PWD" --dependency="afterok:$prep" \
  --kill-on-invalid-dep=yes "$slurm/corrections_sweep.sbatch")
sweep=${sweep%%;*}
printf 'Correction array: %s\n' "$sweep"
report=$(sbatch --parsable --chdir="$PWD" --dependency="afterany:$prep:$sweep" \
  "$slurm/corrections_report.sbatch")
report=${report%%;*}
printf 'Dependent report: %s\n' "$report"
"$WCSP_ENV/bin/python" - "$prep" "$sweep" "$report" <<'PY'
import json,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path('data/fullres_corrections_20261122')
root.mkdir(exist_ok=True)
path=root/'submissions.jsonl'
with path.open('a') as stream:
 stream.write(json.dumps(dict(created=datetime.now(timezone.utc).isoformat(),cluster='discovery',
     prepare=sys.argv[1],sweep=sys.argv[2],report=sys.argv[3]))+'\n')
PY
