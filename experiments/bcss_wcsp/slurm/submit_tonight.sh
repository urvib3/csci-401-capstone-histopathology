#!/bin/bash
# Run from the repository root after downloading data and validating the GPU.
# Set TRAIN_JOB to reuse an existing training array. DRY_RUN=1 prints commands.
set -euo pipefail
mkdir -p data/logs
export WCSP_DATA="${WCSP_DATA:-$PWD/data}"
export WCSP_OUT="${WCSP_OUT:-$PWD/data/tonight}"
export WCSP_ENV="${WCSP_ENV:-$PWD/.venv}"
export DEV_LIMIT="${DEV_LIMIT:-200}"
export PREDICT_SHARDS="${PREDICT_SHARDS:-8}"
submit() {
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'sbatch' >&2; printf ' %q' "$@" >&2; printf '\n' >&2
    echo dryjob
  else
    sbatch --parsable "$@"
  fi
}
if [[ ${DRY_RUN:-0} != 1 ]]; then
  "$WCSP_ENV/bin/python" - <<'CHECK'
import json, os
from pathlib import Path
root = Path(os.environ.get("BCSS_ROOT", str(Path.cwd() / "data/bcss")))
manifest = json.loads((root / "manifest.json").read_text())
assert manifest.get("preparation_version") == 2, "Rebuild masks with current preparation script"
assert all(r.get("preparation_version") == 2 for r in manifest["rois"])
assert len(manifest["rois"]) == 151, "BCSS preparation is incomplete"
assert Path("data/tiatoolbox/models/fcn_resnet50_unet-bcss.pth").is_file(), "Run GPU smoke first to cache weights"
CHECK
fi
slurm=experiments/bcss_wcsp/slurm
configs=experiments/bcss_wcsp/configs/tonight
train=${TRAIN_JOB:-}
if [[ -z "$train" ]]; then
  train=$(submit --array=0-4%3 "$slurm/train.sbatch")
fi
printf 'Training array: %s\n' "$train" >&2
arms=(ce full512 pairwise_0.01 pairwise_0.1 pairwise_1 pretrained ce_full512 pretrained_full512)
training_tasks=(0 1 2 3 4 none 0 none)
for i in "${!arms[@]}"; do
  arm=${arms[$i]}
  dependency=()
  if [[ ${training_tasks[$i]} != none ]]; then dependency=("--dependency=afterok:${train}_${training_tasks[$i]}"); fi
  predict=$(submit "${dependency[@]}" --array="0-$((PREDICT_SHARDS-1))%2" \
    "$slurm/tonight_predict.sbatch" "$configs/$arm.yaml")
  solve=$(submit --dependency="afterok:$predict" --array=0-1 "$slurm/tonight_solve.sbatch" \
    "$configs/$arm.yaml" "tonight_${arm}")
  printf '%s predict=%s solve=%s\n' "$arm" "$predict" "$solve" >&2
  # Long relations and solver comparisons isolate changes on pretrained unary.
  if [[ "$arm" == pretrained ]]; then
    submit --dependency="afterok:$predict" --array=0-1 "$slurm/tonight_solve.sbatch" \
      "$configs/$arm.yaml" tonight_long_control \
      --set 'solver={name: icm, max_sweeps: 50}' >&2
    for d in 8 32; do
      for p in 0.000001 0.00001; do
        submit --dependency="afterok:$predict" --array=0-1 "$slurm/tonight_solve.sbatch" \
          "$configs/$arm.yaml" "tonight_long_d${d}_p${p}" \
          --set "problem.long_distance=$d" --set "problem.long_probability=$p" \
          --set 'solver={name: icm, max_sweeps: 50}' >&2
      done
    done
    for solver in icm toulbar2 hybrid; do
      case "$solver" in
        icm) inner='{name: icm, max_sweeps: 50}' ;;
        toulbar2) inner='{name: toulbar2, time_limit: 2}' ;;
        hybrid) inner='{name: chain, stages: [{name: icm, max_sweeps: 50}, {name: toulbar2, time_limit: 2}]}' ;;
      esac
      submit --dependency="afterok:$predict" --array=0-1 "$slurm/tonight_solve.sbatch" \
        "$configs/$arm.yaml" "tonight_solver_${solver}" --set "solver.inner=$inner" >&2
    done
  fi
done
