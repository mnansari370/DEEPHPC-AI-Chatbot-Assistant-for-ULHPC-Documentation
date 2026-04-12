#!/bin/bash -l
#SBATCH --job-name=deephpc_ft_eval
#SBATCH --output=logs/ft_eval_%j.out
#SBATCH --error=logs/ft_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=volta:1
#SBATCH --mem=40G
#SBATCH --time=01:30:00

# =============================================================================
# DEEPHPC: Fine-Tuned Model Evaluation
# Submit with: sbatch slurm/06_finetune_evaluate.slurm
# =============================================================================

echo "=============================="
echo "DEEPHPC Fine-Tuning Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================="

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

python scripts/run_finetune.py \
    --config configs/finetune_config.yaml \
    --mode evaluate \
    --test-queries evaluation/test_queries.json \
    --output outputs/finetune/results.json

echo "Evaluation complete."
