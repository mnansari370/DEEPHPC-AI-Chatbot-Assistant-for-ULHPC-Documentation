#!/bin/bash -l
#SBATCH --job-name=deephpc_rag_gs
#SBATCH --output=logs/rag_grid_%j.out
#SBATCH --error=logs/rag_grid_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=volta:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# =============================================================================
# DEEPHPC: RAG FAISS Grid Search
# Finds optimal nlist/nprobe hyperparameters.
# Submit with: sbatch slurm/03_rag_grid_search.slurm
# =============================================================================

echo "=============================="
echo "DEEPHPC RAG Grid Search"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "=============================="

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

python scripts/run_rag.py \
    --config configs/rag_config.yaml \
    --mode grid_search \
    --test-queries evaluation/test_queries.json \
    --output outputs/rag/grid_search.json

echo "Grid search complete."
