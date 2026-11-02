#!/bin/bash
#SBATCH --account=jessetho_1732
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=bcss-train
#SBATCH --output=data/logs/%x_%A_%a.out
# Array 0: matched CE baseline; 1: full 512 output; 2-4: pairwise-only arms.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${WCSP_ENV:-$PWD/.venv}/bin/activate"
export TIATOOLBOX_HOME="${TIATOOLBOX_HOME:-$PWD/data/tiatoolbox}"
export HF_HOME="${HF_HOME:-$PWD/data/huggingface}"
export MPLCONFIGDIR="$PWD/data/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
arm=${SLURM_ARRAY_TASK_ID:-0}
case "$arm" in
  0) name=ce; architecture=baseline; field=center; weight=0 ;;
  1) name=full512; architecture=fullres; field=full; weight=0 ;;
  2) name=pairwise_0.01; architecture=baseline; field=center; weight=0.01 ;;
  3) name=pairwise_0.1; architecture=baseline; field=center; weight=0.1 ;;
  4) name=pairwise_1; architecture=baseline; field=center; weight=1 ;;
  *) echo "Unknown training arm $arm" >&2; exit 1 ;;
esac
python -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable"; print(torch.cuda.get_device_name())'
python -u benchmarks/bcss/scripts/train_experiments.py \
  --architecture "$architecture" --field "$field" --pairwise-weight "$weight" \
  --weights "${BCSS_WEIGHTS:-$PWD/data/tiatoolbox/models/fcn_resnet50_unet-bcss.pth}" \
  --root "${BCSS_ROOT:-$PWD/data/bcss}" --out "${TRAIN_OUT_ROOT:-$PWD/data/outputs/tonight}/$name" \
  --epochs "${EPOCHS:-10}" --batch "${BATCH:-4}" --workers 8 --device cuda "$@"
