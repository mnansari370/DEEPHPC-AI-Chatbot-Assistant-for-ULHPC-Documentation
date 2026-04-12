#!/bin/bash -l
#SBATCH --job-name=deephpc_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# =============================================================================
# DEEPHPC: Environment Setup Job
# Submit with: sbatch slurm/00_setup_env.slurm
# =============================================================================

echo "=============================="
echo "DEEPHPC Environment Setup"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "=============================="

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

bash setup.sh
