#!/bin/bash -l
#SBATCH --job-name=restore_env
#SBATCH --output=logs/restore_env_%j.out
#SBATCH --error=logs/restore_env_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=volta:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# =============================================================================
# Full environment restoration.
#
# What went wrong:
#   - fix_bnb.slurm upgraded bitsandbytes to 0.49.2, which pulled torch 2.11.0
#     (requires CUDA 13.0) and triton 3.6.0 (removed triton.ops API)
#   - fix_packages.slurm only partially fixed things
#
# Correct approach:
#   - Restore EVERYTHING pinned to original working versions
#   - Then set BNB_CUDA_VERSION=120 so bitsandbytes 0.43.1 uses CUDA 12 libs
#     instead of looking for the non-existent libcusparse.so.11
# =============================================================================

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

echo "[1/5] Restoring PyTorch 2.1.2+cu118..."
pip install --force-reinstall --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "[2/5] Restoring triton 2.1.0 (bitsandbytes 0.43.1 needs triton.ops API)..."
pip install --force-reinstall --no-cache-dir triton==2.1.0

echo "[3/5] Restoring bitsandbytes 0.43.1..."
pip install --force-reinstall --no-cache-dir bitsandbytes==0.43.1

echo "[4/5] Restoring fsspec and numpy to compatible versions..."
pip install --force-reinstall --no-cache-dir "fsspec[http]==2024.3.1"
pip install --force-reinstall --no-cache-dir numpy==1.26.4

echo "[5/5] Verifying..."
export BNB_CUDA_VERSION=120
python -c "
import torch
print(f'torch:        {torch.__version__}')
print(f'CUDA visible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:          {torch.cuda.get_device_name(0)}')

import bitsandbytes as bnb
print(f'bitsandbytes: {bnb.__version__}')

from peft import LoraConfig
print('peft:         OK')

import datasets
print(f'datasets:     {datasets.__version__}')
"

echo "Environment restored successfully."
