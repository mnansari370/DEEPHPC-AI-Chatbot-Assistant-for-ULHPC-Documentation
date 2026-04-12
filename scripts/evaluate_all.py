"""
DEEPHPC: Side-by-side comparison of RAG vs Fine-Tuned model.

Produces a comparison table and saves results to JSON.

Usage:
    python scripts/evaluate_all.py \
        --rag-config configs/rag_config.yaml \
        --ft-config  configs/finetune_config.yaml \
        --test-queries evaluation/test_queries.json \
        --output-dir outputs/comparison
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.data.prepare_dataset import DatasetPreparer
from src.finetune.inference import FineTunedModel
from src.utils.metrics import compute_all_metrics, print_metrics_table
from src.utils.logging_utils import get_logger

logger = get_logger("EvaluateAll")


def main():
    parser = argparse.ArgumentParser(description="Compare RAG vs Fine-Tuning")
    parser.add_argument("--rag-config",    required=True)
    parser.add_argument("--ft-config",     required=True)
    parser.add_argument("--test-queries",  default="evaluation/test_queries.json")
    parser.add_argument("--output-dir",    default="outputs/comparison")
    parser.add_argument("--index-dir",     default="outputs/rag/index")
    parser.add_argument("--skip-rag",      action="store_true")
    parser.add_argument("--skip-ft",       action="store_true")
    args = parser.parse_args()

    with open(args.rag_config) as f:
        rag_config = yaml.safe_load(f)
    with open(args.ft_config) as f:
        ft_config = yaml.safe_load(f)
    with open(args.test_queries) as f:
        test_data = json.load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    references = [d["answer"] for d in test_data]
    questions  = [d["question"] for d in test_data]

    all_results = {}

    # ── RAG Evaluation ────────────────────────────────────────────────────────
    if not args.skip_rag:
        logger.info("Running RAG evaluation...")
        pipeline = RAGPipeline(rag_config)
        preparer = DatasetPreparer({**rag_config, **ft_config})
        chunks   = preparer.load_raw_chunks()
        pipeline.build_or_load_index(chunks, args.index_dir)

        rag_predictions = []
        for q in questions:
            result = pipeline.query(q, with_generation=True)
            rag_predictions.append(result["answer"] or " ".join(result["context"]))

        rag_metrics = compute_all_metrics(rag_predictions, references)
        all_results["rag"] = {
            "metrics":     rag_metrics,
            "predictions": rag_predictions,
        }
        print_metrics_table(rag_metrics, model_name="RAG (Hybrid FAISS+BM25)")

    # ── Fine-Tuned Evaluation ─────────────────────────────────────────────────
    if not args.skip_ft:
        logger.info("Running fine-tuned model evaluation...")
        ft_model    = FineTunedModel(ft_config)
        ft_preds    = [ft_model.answer(q) for q in questions]
        ft_metrics  = compute_all_metrics(ft_preds, references)
        all_results["finetune"] = {
            "metrics":     ft_metrics,
            "predictions": ft_preds,
        }
        print_metrics_table(ft_metrics, model_name="Fine-Tuned (QLoRA DeepSeek)")

    # ── Comparison table ──────────────────────────────────────────────────────
    if "rag" in all_results and "finetune" in all_results:
        print("\n" + "="*65)
        print(f"{'Metric':<30} {'RAG':>10} {'Fine-Tuned':>12} {'Winner':>10}")
        print("-"*65)
        for metric in ["cosine_similarity", "rouge_l", "rouge_1"]:
            rag_score = all_results["rag"]["metrics"].get(metric, {}).get("mean", 0.0)
            ft_score  = all_results["finetune"]["metrics"].get(metric, {}).get("mean", 0.0)
            winner    = "RAG" if rag_score > ft_score else "Fine-Tuned"
            print(f"{metric:<30} {rag_score:>10.4f} {ft_score:>12.4f} {winner:>10}")
        print("="*65)

    # Save full comparison
    output_path = Path(args.output_dir) / "comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
