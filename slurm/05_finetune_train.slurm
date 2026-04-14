#!/bin/bash -l
#SBATCH --job-name=deephpc_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=volta:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00

# =============================================================================
# DEEPHPC: QLoRA Fine-Tuning Job
# Fine-tunes DeepSeek-R1-Distill-Qwen-1.5B on ULHPC Q&A data.
# Submit with: sbatch slurm/05_finetune_train.slurm
#
# For better GPU (recommended for faster training):
#   Change --partition=gpu to --partition=hopper or --partition=l40s
#   Change --gpus-per-node=volta:1 to --gpus-per-node=hopper:1 or l40s:1
# =============================================================================

echo "=============================="
echo "DEEPHPC QLoRA Fine-Tuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================="

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

# Tell bitsandbytes to use CUDA 12 libraries (nodes have CUDA 12.7 driver,
# libcusparse.so.12 not .11)
export BNB_CUDA_VERSION=120

# Log GPU info
nvidia-smi

python scripts/run_finetune.py \
    --config configs/finetune_config.yaml \
    --mode train \
    --train-data data/qa_train.json \
    --val-data data/qa_val.json

echo "Fine-tuning complete."
