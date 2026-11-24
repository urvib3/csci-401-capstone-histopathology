#!/bin/bash
# Build the BiomedParse v1 environment (2D pathology weights need v1 + its detectron2 fork,
# which compiles against CUDA, so this runs on a GPU node).
#
#   sbatch --clusters=discovery experiments/bcss_wcsp/slurm/biomedparse_env.sbatch
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=bmp1-env
#SBATCH --output=/scratch1/%u/wcsp_logs/%x_%j.out

set -euo pipefail
export TMPDIR=/scratch1/$USER/tmp PIP_CACHE_DIR=/scratch1/$USER/tmp/pipcache CONDA_PKGS_DIRS=/scratch1/$USER/tmp/conda_pkgs
mkdir -p "$TMPDIR"
ENV=/scratch1/$USER/envs/bmp1
SRC=/scratch1/$USER/vlm/BiomedParse_v1

module purge
module load conda/25.11.0
[ -d "$ENV" ] || mamba create -y -q --prefix "$ENV" python=3.9.19
PY=$ENV/bin/python

module load gcc/13.3.0 cuda/12.6.3 openmpi/5.0.5   # cuda 12.6 pairs with gcc 13.3; torch cu124 wheels build against it
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;9.0" MAX_JOBS=4

$PY -m pip install -q --upgrade pip
$PY -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
# requirements.txt without the git detectron2 line, then the fork without build isolation (needs torch)
grep -v "detectron2" "$SRC/assets/requirements/requirements.txt" > "$TMPDIR/bmp1_req.txt"
$PY -m pip install -q -r "$TMPDIR/bmp1_req.txt"
$PY -m pip install -q --no-build-isolation "git+https://github.com/MaureenZOU/detectron2-xyz.git"
$PY -m pip install -q "huggingface_hub<1.0" scipy

cd "$SRC"
$PY -c "import torch, detectron2; from modeling.BaseModel import BaseModel; print('ok', torch.__version__, torch.cuda.is_available())"
