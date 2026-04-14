#!/bin/bash -l
#SBATCH --job-name=fix_packages
#SBATCH --output=logs/fix_packages_%j.out
#SBATCH --error=logs/fix_packages_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=volta:1
#SBATCH --mem=8G
#SBATCH --time=00:20:00

# =============================================================================
# Restore environment after the bad bitsandbytes upgrade pulled torch 2.11
# which requires CUDA 13 — our driver only supports CUDA 12.7.
#
# Strategy:
#   1. Restore torch 2.1.2+cu118  (works with CUDA 12.7 driver — backward compat)
#   2. Install bitsandbytes 0.44.1 (supports CUDA 12, compatible with torch 2.1.x)
#   3. Restore fsspec to datasets-compatible version
#   4. Verify CUDA is visible
# =============================================================================

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

echo "[1/4] Restoring torch 2.1.2+cu118..."
pip install --force-reinstall --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "[2/4] Installing bitsandbytes 0.44.1 (CUDA 12 compatible, torch>=2.0)..."
pip install --force-reinstall --no-cache-dir bitsandbytes==0.44.1

echo "[3/4] Restoring fsspec to datasets-compatible version..."
pip install --force-reinstall --no-cache-dir "fsspec[http]==2024.3.1"

echo "[4/4] Verifying..."
python -c "
import torch
print(f'torch:        {torch.__version__}')
print(f'CUDA visible: {torch.cuda.is_available()}')
print(f'GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

import bitsandbytes as bnb
print(f'bitsandbytes: {bnb.__version__}')
"

echo "Done."
