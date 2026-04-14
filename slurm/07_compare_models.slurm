#!/bin/bash -l
#SBATCH --job-name=deephpc_compare
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=volta:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# =============================================================================
# DEEPHPC: Compare RAG vs Fine-Tuned Model
# Runs both and produces a side-by-side comparison report.
# Submit with: sbatch slurm/07_compare_models.slurm
# =============================================================================

echo "=============================="
echo "DEEPHPC Model Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================="

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

export BNB_CUDA_VERSION=120

# faiss-cpu 1.8.0 was compiled against numpy 1.x; the bitsandbytes upgrade
# chain pulled numpy 2.2.6 which breaks faiss. Pin back to original version.
pip install -q numpy==1.26.4 --force-reinstall

python scripts/evaluate_all.py \
    --rag-config configs/rag_config.yaml \
    --ft-config configs/finetune_config.yaml \
    --test-queries evaluation/test_queries.json \
    --output-dir outputs/comparison

echo "Comparison complete. Results in outputs/comparison/"
