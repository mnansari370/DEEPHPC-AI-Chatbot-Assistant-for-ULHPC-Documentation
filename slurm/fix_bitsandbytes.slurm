#!/bin/bash -l
#SBATCH --job-name=fix_bnb
#SBATCH --output=logs/fix_bnb_%j.out
#SBATCH --error=logs/fix_bnb_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=volta:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00

# =============================================================================
# Fix bitsandbytes CUDA version mismatch.
# bitsandbytes 0.43.1 was built for CUDA 11.x but the nodes have CUDA 12.
# Upgrade to a version with proper auto-detection.
# =============================================================================

cd $SLURM_SUBMIT_DIR
source /home/users/nmo/miniconda3/etc/profile.d/conda.sh
conda activate ULHPC_env

echo "Current bitsandbytes version:"
pip show bitsandbytes | grep Version

echo "Upgrading bitsandbytes..."
pip install bitsandbytes --upgrade --force-reinstall

echo "Verifying fix:"
python -m bitsandbytes

echo "Done."
