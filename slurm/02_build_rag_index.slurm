#!/bin/bash -l
#SBATCH --job-name=deephpc_rag_index
#SBATCH --output=logs/rag_index_%j.out
#SBATCH --error=logs/rag_index_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# =============================================================================
# DEEPHPC: RAG Index Building Job
# Embeds all chunks and builds FAISS + BM25 indexes.
# Submit with: sbatch slurm/02_build_rag_index.slurm
# =============================================================================

echo "=============================="
echo "DEEPHPC RAG Index Building"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================="

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

python scripts/run_rag.py \
    --config configs/rag_config.yaml \
    --mode build_index

echo "Index building complete."
