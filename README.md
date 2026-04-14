# DEEPHPC: AI Chatbot Assistant for ULHPC Documentation

> **A comparative study of Retrieval-Augmented Generation (RAG) vs. LoRA Fine-Tuning for domain-specific LLM adaptation on High-Performance Computing documentation.**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

DEEPHPC is an AI-powered chatbot that answers natural language questions about the [University of Luxembourg HPC (ULHPC)](https://hpc.uni.lu) cluster. The system adapts **DeepSeek-R1-Distill-Qwen-1.5B** using two complementary strategies and compares them end-to-end:

| Approach | Method | Key Strength |
|---|---|---|
| **RAG** | Hybrid BM25 + FAISS retrieval → DeepSeek generation | No retraining, always up-to-date |
| **Fine-Tuning** | LoRA (fp16) on curated Q&A pairs | Higher answer precision, offline use |

The entire pipeline runs on the **ULHPC cluster via SLURM**, making it a practical end-to-end example of running LLM workloads on HPC infrastructure.

---

## Results

### Final Comparison — RAG vs Fine-Tuning

| Metric | RAG (FAISS+BM25) | Fine-Tuned (LoRA) | Winner |
|---|---|---|---|
| Cosine Similarity (TF-IDF) | 0.2816 | **0.2900** | Fine-Tuned (+3.0%) |
| ROUGE-L F1 | 0.1487 | **0.1986** | Fine-Tuned (+33.6%) |
| ROUGE-1 F1 | 0.1958 | **0.2652** | Fine-Tuned (+35.4%) |

Fine-tuning outperforms RAG on all metrics. The ROUGE improvement shows the fine-tuned model learned ULHPC-specific vocabulary and answer structure from the training data.

### FAISS Grid Search Results

| Config | Accuracy | Avg Time | Fitness |
|---|---|---|---|
| nlist=20, nprobe=1 ✓ | 0.1542 | 0.0114s | **0.1045** (best) |
| nlist=1, nprobe=20 | 0.1387 | 0.0096s | 0.0942 |
| nlist=10, nprobe=5 | 0.1340 | 0.0108s | 0.0906 |

Best config: **nlist=20, nprobe=1** (highest fitness balancing accuracy and speed).

### LoRA Fine-Tuning Training Curve

| Epoch | Train Loss | Eval Loss |
|---|---|---|
| 1 | 2.6494 | 2.0009 |
| 2 | 1.6943 | 1.5406 |
| 3 | 1.4976 | **1.4680** |

- Trainable parameters: 18.5M / 1.8B (1.03%)
- Training time: ~3.35 hours on NVIDIA Tesla V100 16GB
- No overfitting — eval loss improves consistently across all 3 epochs

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
│    ┌────────┴────────┐                              │
│    ▼                 ▼                              │
│  FAISS IVF        BM25 (Okapi)                     │
│  (dense)          (sparse)                          │
│    └────────┬────────┘                              │
│             ▼                                       │
│     RRF Score Fusion (top-5 chunks)                │
│             │                                       │
│             ▼                                       │
│  DeepSeek-R1-Distill-Qwen-1.5B (Generator)         │
│             ▼                                       │
│          Answer                                     │
└─────────────────────────────────────────────────────┘

      ──────── OR ────────

┌─────────────────────────────────────────────────────┐
│              Fine-Tuning Pipeline                   │
│                                                     │
│  ULHPC Markdown Docs → Sentence-Aware Chunking     │
│       → Q&A Generation (10 templates)              │
│       → Tokenization (512 tokens max)              │
│       → LoRA Fine-Tuning (r=16, α=32, fp16)       │
│       → Fine-Tuned DeepSeek-R1 (offline)           │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
deephpc/
├── configs/
│   ├── rag_config.yaml           # RAG hyperparameters (nlist=20, nprobe=1)
│   └── finetune_config.yaml      # LoRA training config (r=16, α=32)
├── src/
│   ├── data/
│   │   └── prepare_dataset.py    # Doc parsing, sentence-aware chunking, Q&A generation
│   ├── rag/
│   │   ├── embedder.py           # SentenceTransformer encoding
│   │   ├── retriever.py          # Hybrid BM25+FAISS with RRF fusion
│   │   ├── generator.py          # DeepSeek response generation
│   │   └── pipeline.py           # End-to-end RAG orchestration
│   ├── finetune/
│   │   ├── dataset.py            # Tokenization & dataset building
│   │   ├── train.py              # LoRA training loop
│   │   └── inference.py          # LoRA adapter loading & inference
│   └── utils/
│       ├── metrics.py            # ROUGE-L, Cosine, BERTScore
│       └── logging_utils.py      # Structured logging
├── scripts/
│   ├── prepare_data.py           # CLI: clone docs + generate dataset
│   ├── run_rag.py                # CLI: build index / evaluate / query
│   ├── run_finetune.py           # CLI: train / evaluate / query
│   └── evaluate_all.py           # CLI: RAG vs FT comparison
├── slurm/
│   ├── 00_setup_env.slurm        # Install dependencies
│   ├── 01_prepare_data.slurm     # Generate dataset
│   ├── 02_build_rag_index.slurm  # Build FAISS+BM25 index
│   ├── 03_rag_grid_search.slurm  # FAISS hyperparameter search
│   ├── 04_rag_evaluate.slurm     # RAG evaluation
│   ├── 05_finetune_train.slurm   # LoRA training (GPU)
│   ├── 06_finetune_evaluate.slurm# Fine-tune evaluation
│   └── 07_compare_models.slurm   # Side-by-side comparison
├── evaluation/
│   └── test_queries.json         # 10 ULHPC ground-truth Q&A pairs
├── data/
│   ├── qa_train.json             # 335 training Q&A pairs (generated)
│   ├── qa_val.json               # 38 validation Q&A pairs (generated)
│   └── qa_dataset.json           # Full dataset (373 pairs)
├── outputs/
│   ├── rag/
│   │   ├── results.json          # RAG evaluation results
│   │   ├── grid_search.json      # FAISS hyperparameter search results
│   │   └── index/                # FAISS + BM25 index files
│   ├── finetune/
│   │   ├── results.json          # Fine-tuned model evaluation results
│   │   └── final_adapter/        # Saved LoRA adapter weights (71MB)
│   └── comparison/
│       └── comparison.json       # Full RAG vs Fine-Tuned comparison
├── setup.sh                      # One-shot environment setup
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites

- ULHPC account with access to GPU partition (Volta V100 or better)
- Miniconda installed at `~/miniconda3`
- Python 3.10

### 2. Environment Setup (run once)

```bash
git clone https://github.com/mnansari370/DEEPHPC-AI-Chatbot-Assistant-for-ULHPC-Documentation.git
cd DEEPHPC-AI-Chatbot-Assistant-for-ULHPC-Documentation

# Create conda environment
conda create -n ULHPC_env python=3.10 -y
conda activate ULHPC_env

# Install all dependencies
bash setup.sh
```

### 3. Run the Full Pipeline via SLURM

```bash
# Step 0: Install dependencies (once, ~10 min)
sbatch slurm/00_setup_env.slurm

# Step 1: Clone ULHPC docs + generate Q&A dataset (~5 min, CPU)
sbatch slurm/01_prepare_data.slurm

# Step 2: Build FAISS + BM25 index (~5 min, CPU)
sbatch slurm/02_build_rag_index.slurm

# Step 3: FAISS hyperparameter grid search (~30 min, GPU)
sbatch slurm/03_rag_grid_search.slurm

# Step 4: RAG evaluation with generation (~60-90 min, GPU)
sbatch slurm/04_rag_evaluate.slurm

# Step 5: LoRA fine-tuning (~3-4h on V100 16GB, GPU)
sbatch slurm/05_finetune_train.slurm

# Step 6: Fine-tuned model evaluation (~30-60 min, GPU)
sbatch slurm/06_finetune_evaluate.slurm

# Step 7: Side-by-side comparison
sbatch slurm/07_compare_models.slurm
```

Monitor jobs: `squeue -u $USER` | Logs: `tail -f logs/<jobname>_<JOBID>.err`

### 4. Ask Questions Interactively

```bash
conda activate ULHPC_env

# Ask the RAG pipeline
python scripts/run_rag.py --config configs/rag_config.yaml \
    --mode query --question "How do I submit a GPU job on ULHPC?"

# Ask the fine-tuned model
python scripts/run_finetune.py --config configs/finetune_config.yaml \
    --mode query --question "How do I submit a GPU job on ULHPC?"
```

---

## Dataset

The dataset is generated automatically from the [ULHPC documentation](https://github.com/ULHPC/ulhpc-docs):

| Split | Pairs | Source |
|---|---|---|
| Train | 335 | `data/qa_train.json` |
| Validation | 38 | `data/qa_val.json` |
| Full | 373 | `data/qa_dataset.json` |

**Generation process:**
1. Clone ULHPC markdown documentation
2. Parse with sentence-aware chunking (400-word chunks, 80-word overlap)
3. Filter by quality (20–150 words, no URLs, low code ratio)
4. Apply one of 10 instruction templates per chunk
5. 90/10 train/val split with deduplication

---

## Model Details

**Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

| Feature | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~1.5 billion |
| Context Window | 4096 tokens |

**LoRA Configuration:**

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Precision | fp16 |
| Optimizer | AdamW |
| Trainable Parameters | 18.5M / 1.8B (1.03%) |

**RAG Configuration:**

| Component | Value |
|---|---|
| Embedder | `all-MiniLM-L6-v2` (384-dim) |
| FAISS Index | IVF (nlist=20, nprobe=1) |
| BM25 | Okapi BM25 (k1=1.5, b=0.75) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Top-K chunks | 5 |
| Dense weight | 0.6 |
| Sparse weight | 0.4 |

---

## Technical Highlights

- **Hybrid Retrieval (RRF):** Combines FAISS dense scores with BM25 sparse scores using Reciprocal Rank Fusion — handles both semantic queries ("tell me about authentication") and keyword queries ("sbatch --gres flag")
- **Sentence-Aware Chunking:** Splits on sentence boundaries rather than fixed character counts, preserving semantic units and reducing fragmentation
- **FAISS Grid Search:** Automated sweep over nlist × nprobe parameter space using a fitness function balancing accuracy and retrieval speed (λ=0.7)
- **LoRA Fine-Tuning:** 1.03% trainable parameters, gradient checkpointing enabled — trains on a 16GB V100 in ~3.5 hours
- **Multi-Metric Evaluation:** ROUGE-L (structural overlap), ROUGE-1 (unigram precision), and TF-IDF Cosine Similarity

---

## GPU Partitions on ULHPC

| Partition | GPU | VRAM | Notes |
|---|---|---|---|
| `gpu` | NVIDIA Volta V100 | 16–32GB | Default, used in this project |
| `hopper` | NVIDIA Hopper H100 | 80GB | Fastest, recommended for larger models |
| `l40s` | NVIDIA L40S | 48GB | Good for inference |

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
3. Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
4. Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain QA.* EMNLP 2020.
5. Cormack & Clarke (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR 2009.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
