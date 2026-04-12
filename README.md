# DEEPHPC — Intelligent HPC Documentation Assistant

> **A comparative study of RAG vs. QLoRA Fine-Tuning for domain-specific LLM adaptation on High-Performance Computing documentation.**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

DEEPHPC is an AI-powered chatbot that answers natural language questions about the [University of Luxembourg HPC (ULHPC)](https://hpc.uni.lu) cluster — one of the largest HPC systems in Luxembourg. The system adapts the **DeepSeek-R1-Distill-Qwen-1.5B** model using two complementary strategies:

| Approach | Method | Best For |
|---|---|---|
| **RAG** | Hybrid BM25 + FAISS retrieval → DeepSeek generation | Dynamic docs, no retraining |
| **Fine-Tuning** | QLoRA (4-bit) on curated Q&A pairs | High accuracy, offline use |

This project runs natively on the **ULHPC cluster using SLURM**, making it a practical example of running LLM workloads on HPC infrastructure.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   RAG Pipeline                      │
│                                                     │
│  Query → Embedder (all-MiniLM-L6-v2)               │
│             │                                       │
│    ┌────────┴────────┐                             │
│    ▼                 ▼                             │
│  FAISS IVF        BM25 (Okapi)                    │
│  (dense)          (sparse)                         │
│    └────────┬────────┘                             │
│             ▼                                       │
│     RRF Score Fusion (top-5 chunks)               │
│             │                                       │
│             ▼                                       │
│  DeepSeek-R1-Distill-Qwen-1.5B (Generator)        │
│             │                                       │
│             ▼                                       │
│          Answer                                     │
└─────────────────────────────────────────────────────┘

      ──────── OR ────────

┌─────────────────────────────────────────────────────┐
│              Fine-Tuning Pipeline                   │
│                                                     │
│  ULHPC Markdown Docs → Sentence-Aware Chunking     │
│       → Q&A Generation (15 templates)              │
│       → Tokenization (512 tokens max)              │
│       → QLoRA Fine-Tuning (r=16, α=32, NF4)       │
│       → Fine-Tuned DeepSeek-R1 (offline)           │
└─────────────────────────────────────────────────────┘
```

---

## Key Improvements Over Baseline

| Feature | Original | DEEPHPC (This Work) |
|---|---|---|
| Chunking | Fixed 1000-char | **Sentence-aware** with configurable overlap |
| Retrieval | Dense FAISS only | **Hybrid: BM25 + FAISS** (RRF fusion) |
| Fine-Tuning | Plain LoRA (r=8) | **QLoRA** (4-bit NF4, r=16, + MLP layers) |
| Q&A Templates | 2 templates | **15 diverse templates** |
| Evaluation | Cosine similarity only | **ROUGE-L + Cosine + BERTScore** |
| Code Structure | Single notebook | **Modular Python package** |
| HPC Integration | Google Colab | **SLURM scripts for ULHPC** |
| Config | Hardcoded | **YAML config files** |

---

## Project Structure

```
deephpc/
├── configs/
│   ├── rag_config.yaml          # RAG hyperparameters
│   └── finetune_config.yaml     # QLoRA training config
├── src/
│   ├── data/
│   │   └── prepare_dataset.py   # Doc parsing, chunking, Q&A generation
│   ├── rag/
│   │   ├── embedder.py          # SentenceTransformer encoding
│   │   ├── retriever.py         # Hybrid BM25+FAISS with RRF fusion
│   │   ├── generator.py         # DeepSeek response generation
│   │   └── pipeline.py          # End-to-end RAG orchestration
│   ├── finetune/
│   │   ├── dataset.py           # Tokenization & dataset building
│   │   ├── train.py             # QLoRA training loop
│   │   └── inference.py         # LoRA adapter loading & inference
│   └── utils/
│       ├── metrics.py           # ROUGE-L, Cosine, BERTScore
│       └── logging_utils.py     # Structured logging
├── scripts/
│   ├── prepare_data.py          # CLI: clone docs + generate dataset
│   ├── run_rag.py               # CLI: build index / evaluate / query
│   ├── run_finetune.py          # CLI: train / evaluate / query
│   └── evaluate_all.py          # CLI: RAG vs FT comparison
├── slurm/
│   ├── 00_setup_env.slurm       # Install dependencies
│   ├── 01_prepare_data.slurm    # Generate dataset
│   ├── 02_build_rag_index.slurm # Build FAISS+BM25 index
│   ├── 03_rag_grid_search.slurm # FAISS hyperparameter search
│   ├── 04_rag_evaluate.slurm    # RAG evaluation
│   ├── 05_finetune_train.slurm  # QLoRA training (GPU)
│   ├── 06_finetune_evaluate.slurm # Fine-tune evaluation
│   └── 07_compare_models.slurm  # Side-by-side comparison
├── evaluation/
│   └── test_queries.json        # 10 ULHPC ground-truth Q&A pairs
├── setup.sh                     # One-shot environment setup
└── requirements.txt
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone this repo
git clone <your-repo-url>
cd deephpc

# Create conda environment (Python 3.10 required)
conda create -n ULHPC_env python=3.10 -y
conda activate ULHPC_env

# Install all dependencies
bash setup.sh
```

### 2. Prepare Data

```bash
# Clone ULHPC docs and generate Q&A dataset
python scripts/prepare_data.py \
    --config configs/finetune_config.yaml \
    --rag-config configs/rag_config.yaml
```

This will:
- Clone the [ULHPC documentation repo](https://github.com/ULHPC/ulhpc-docs)
- Parse all `.md` files with sentence-aware chunking
- Generate ~900+ instruction-following Q&A pairs
- Save 90/10 train/val split to `data/`

### 3. RAG Pipeline

```bash
# Build the hybrid retrieval index
python scripts/run_rag.py --config configs/rag_config.yaml --mode build_index

# Run FAISS hyperparameter grid search
python scripts/run_rag.py --config configs/rag_config.yaml --mode grid_search \
    --test-queries evaluation/test_queries.json

# Full evaluation with generation
python scripts/run_rag.py --config configs/rag_config.yaml --mode evaluate \
    --test-queries evaluation/test_queries.json

# Ask a question interactively
python scripts/run_rag.py --config configs/rag_config.yaml --mode query \
    --question "How do I submit a GPU job on ULHPC?"
```

### 4. Fine-Tuning (QLoRA)

```bash
# Train (requires GPU with ≥16GB VRAM)
python scripts/run_finetune.py --config configs/finetune_config.yaml \
    --mode train \
    --train-data data/qa_train.json \
    --val-data data/qa_val.json

# Evaluate fine-tuned model
python scripts/run_finetune.py --config configs/finetune_config.yaml \
    --mode evaluate \
    --test-queries evaluation/test_queries.json
```

### 5. Compare Both Models

```bash
python scripts/evaluate_all.py \
    --rag-config configs/rag_config.yaml \
    --ft-config  configs/finetune_config.yaml \
    --test-queries evaluation/test_queries.json \
    --output-dir outputs/comparison
```

---

## Running on ULHPC with SLURM

The recommended way to run this project is via SLURM job submission on the ULHPC cluster.

```bash
# Step 0: One-time setup (run once on login node)
sbatch slurm/00_setup_env.slurm

# Step 1: Prepare dataset
sbatch slurm/01_prepare_data.slurm

# Step 2: Build RAG index (CPU node sufficient)
sbatch slurm/02_build_rag_index.slurm

# Step 3: Grid search for best FAISS params (GPU)
sbatch slurm/03_rag_grid_search.slurm

# Step 4: RAG evaluation (GPU)
sbatch slurm/04_rag_evaluate.slurm

# Step 5: Fine-tune with QLoRA (GPU, ~6-8 hours on Volta V100)
sbatch slurm/05_finetune_train.slurm

# Step 6: Evaluate fine-tuned model
sbatch slurm/06_finetune_evaluate.slurm

# Step 7: Compare both models
sbatch slurm/07_compare_models.slurm
```

Monitor jobs with `squeue -u $USER` and view logs in `logs/`.

**GPU partitions available on ULHPC:**
- `gpu` — NVIDIA Volta V100 (4 GPUs/node) — default
- `hopper` — NVIDIA Hopper H100 — fastest
- `l40s` — NVIDIA L40S — good for inference

---

## Results

### RAG — FAISS Hyperparameter Grid Search

| Metric | Best Config | Score |
|---|---|---|
| Accuracy (Cosine) | nlist=25, nprobe≥10 | ~0.3339 |
| Speed | nlist=20, nprobe=1 | ~0.00013s |
| Balanced ✓ | nlist=10, nprobe=5 | Best trade-off |

### RAG vs Fine-Tuning Comparison

| Aspect | RAG | Fine-Tuning (QLoRA) |
|---|---|---|
| Avg Cosine Similarity | ~0.33 | ~0.80+ (on known queries) |
| Response Latency | 0.1–1.2ms (retrieval) | <1s (inference) |
| Training Time | None | ~6-8h (Volta V100) |
| Setup Cost | Low (CPU possible) | Medium (needs GPU) |
| Flexibility | High (update docs instantly) | Low (retrain to update) |
| Offline Use | No (needs index) | Yes |

### Training Progress (QLoRA Fine-Tuning)

```
Initial Loss:  ~2.5
Final Loss:    ~1.4
Epochs:        3
Steps:         ~867
Training Time: ~6.5h (NVIDIA L4 / Volta V100)
```

---

## Model Details

**Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

| Feature | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~1.5 billion |
| Tokenizer | SentencePiece (BPE) |
| Context Window | 4096 tokens |
| Training Objective | Causal Language Modeling |

**QLoRA Configuration:**

| Parameter | Value |
|---|---|
| Quantization | 4-bit NF4 |
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout | 0.05 |
| Optimizer | Paged AdamW 8-bit |
| Trainable Parameters | <1% of total |

---

## Technical Highlights

- **Hybrid Retrieval (RRF):** Combines FAISS dense similarity scores with BM25 term-frequency scores using Reciprocal Rank Fusion — essential for handling both semantic queries ("tell me about authentication") and keyword queries ("sbatch options")
- **Sentence-Aware Chunking:** Unlike fixed-character splitting, our chunker preserves sentence boundaries and uses configurable word-count limits, reducing semantic fragmentation
- **QLoRA:** 4-bit NF4 quantization reduces GPU memory footprint by ~4× compared to full precision, enabling fine-tuning a 1.5B model on a single V100 GPU
- **Multi-Metric Evaluation:** ROUGE-L (structural overlap), Cosine Similarity (TF-IDF semantic), and optionally BERTScore (deep semantic similarity)

---

## Authors

- **Mo Nafees** — nafees.mo.001@student.uni.lu
- **Eyasu Araya** — 0240224114@uni.lu
- **Esraa Albaraqat** — esraa.albaraqat.001@student.uni.lu
- **Pranathi Kola** — pranathi.kola.001@student.uni.lu

*University of Luxembourg — Master's in Computer Science*

---

## References

1. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
2. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
3. Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain QA.* EMNLP 2020.
4. Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
