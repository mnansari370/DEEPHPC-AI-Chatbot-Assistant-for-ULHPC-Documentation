#!/bin/bash -l
#SBATCH --job-name=deephpc_rag_eval
#SBATCH --output=logs/rag_eval_%j.out
#SBATCH --error=logs/rag_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=volta:1
#SBATCH --mem=40G
#SBATCH --time=01:30:00

# =============================================================================
# DEEPHPC: RAG Full Evaluation
# Runs end-to-end RAG with generation and evaluation.
# Submit with: sbatch slurm/04_rag_evaluate.slurm
# =============================================================================

echo "=============================="
echo "DEEPHPC RAG Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================="

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

python scripts/run_rag.py \
    --config configs/rag_config.yaml \
    --mode evaluate \
    --test-queries evaluation/test_queries.json \
    --output outputs/rag/results.json

echo "RAG evaluation complete."
