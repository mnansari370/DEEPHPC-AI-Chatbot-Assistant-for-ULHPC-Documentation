#!/bin/bash
# =============================================================================
# DEEPHPC Environment Setup Script
# Run this ONCE on a login node to set up the conda environment
# =============================================================================

set -e

ENV_NAME="ULHPC_env"
CONDA_BASE="/home/users/nmo/miniconda3"

echo "============================================="
echo "  DEEPHPC Environment Setup"
echo "============================================="

# Source conda
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate existing env (already created with Python 3.10)
conda activate ${ENV_NAME}

echo "[1/4] Installing PyTorch (CUDA 11.8)..."
pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "[2/4] Installing HuggingFace ecosystem..."
pip install --no-cache-dir \
    transformers==4.41.1 \
    tokenizers==0.19.1 \
    datasets==2.19.1 \
    peft==0.10.0 \
    accelerate==0.30.1 \
    bitsandbytes==0.43.1

echo "[3/4] Installing RAG & evaluation libraries..."
pip install --no-cache-dir \
    faiss-cpu==1.8.0 \
    sentence-transformers==2.7.0 \
    rank-bm25==0.2.2 \
    rouge-score==0.1.2 \
    bert-score==0.3.13 \
    scikit-learn==1.5.0 \
    nltk==3.8.1

echo "[4/4] Installing utilities..."
pip install --no-cache-dir \
    markdown==3.6 \
    beautifulsoup4==4.12.3 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    rich==13.7.1 \
    loguru==0.7.2 \
    matplotlib==3.9.0 \
    seaborn==0.13.2

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo ""
echo "============================================="
echo "  Setup complete! Activate with:"
echo "  conda activate ${ENV_NAME}"
echo "============================================="
