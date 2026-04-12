#!/bin/bash -l
#SBATCH --job-name=deephpc_data
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# =============================================================================
# DEEPHPC: Data Preparation Job
# Clones ULHPC docs and generates Q&A dataset.
# Submit with: sbatch slurm/01_prepare_data.slurm
# =============================================================================

echo "=============================="
echo "DEEPHPC Data Preparation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "=============================="

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

python scripts/prepare_data.py \
    --config configs/finetune_config.yaml \
    --rag-config configs/rag_config.yaml

echo "Data preparation complete."
