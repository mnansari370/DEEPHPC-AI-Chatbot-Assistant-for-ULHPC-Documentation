"""
DEEPHPC RAG Script.

Modes:
  build_index  — Build FAISS + BM25 indexes from documentation
  grid_search  — Optimize FAISS nlist/nprobe hyperparameters
  evaluate     — Run full RAG evaluation with generation
  query        — Answer a single interactive question

Usage:
    python scripts/run_rag.py --config configs/rag_config.yaml --mode build_index
    python scripts/run_rag.py --config configs/rag_config.yaml --mode grid_search \
        --test-queries evaluation/test_queries.json
    python scripts/run_rag.py --config configs/rag_config.yaml --mode evaluate \
        --test-queries evaluation/test_queries.json
    python scripts/run_rag.py --config configs/rag_config.yaml --mode query \
        --question "How do I submit a GPU job?"
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.data.prepare_dataset import DatasetPreparer
from src.utils.logging_utils import get_logger

logger = get_logger("RunRAG")


def main():
    parser = argparse.ArgumentParser(description="DEEPHPC RAG Pipeline")
    parser.add_argument("--config",        required=True, help="RAG config YAML")
    parser.add_argument("--mode",          required=True,
                        choices=["build_index", "grid_search", "evaluate", "query"])
    parser.add_argument("--test-queries",  default="evaluation/test_queries.json")
    parser.add_argument("--output",        default=None)
    parser.add_argument("--question",      default=None, help="Question for query mode")
    parser.add_argument("--index-dir",     default="outputs/rag/index")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = RAGPipeline(config)

    # ── Build index ──────────────────────────────────────────────────────────
    if args.mode == "build_index":
        preparer = DatasetPreparer(config)
        chunks   = preparer.load_raw_chunks()
        pipeline.build_or_load_index(chunks, args.index_dir)
        logger.info(f"Index stored at: {args.index_dir}")

    # ── Grid search ──────────────────────────────────────────────────────────
    elif args.mode == "grid_search":
        # Load index first
        preparer = DatasetPreparer(config)
        chunks   = preparer.load_raw_chunks()
        pipeline.build_or_load_index(chunks, args.index_dir)

        output = args.output or "outputs/rag/grid_search.json"
        results = pipeline.run_grid_search(args.test_queries, output)
        logger.info(f"Best: nlist={results['best_nlist']}, nprobe={results['best_nprobe']}, "
                    f"fitness={results['best_fitness']:.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    elif args.mode == "evaluate":
        preparer = DatasetPreparer(config)
        chunks   = preparer.load_raw_chunks()
        pipeline.build_or_load_index(chunks, args.index_dir)

        output = args.output or config["evaluation"]["output_path"]
        metrics = pipeline.evaluate(args.test_queries, output)
        logger.info(f"ROUGE-L: {metrics['rouge_l']['mean']:.4f}")
        logger.info(f"Cosine:  {metrics['cosine_similarity']['mean']:.4f}")

    # ── Interactive query ─────────────────────────────────────────────────────
    elif args.mode == "query":
        pipeline.retriever.load_index(args.index_dir)

        question = args.question or input("Enter your question: ")
        result   = pipeline.query(question, with_generation=True)

        print("\n" + "="*60)
        print(f"QUESTION: {result['question']}")
        print("-"*60)
        print(f"RETRIEVED CONTEXT ({len(result['context'])} chunks):")
        for i, chunk in enumerate(result['context'], 1):
            print(f"  [{i}] {chunk[:120]}...")
        print("-"*60)
        print(f"ANSWER:\n{result['answer']}")
        print(f"\nSearch time: {result['search_time']*1000:.2f}ms")
        print("="*60)


if __name__ == "__main__":
    main()
