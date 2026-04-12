"""
DEEPHPC RAG Pipeline — end-to-end orchestration.

Ties together: DatasetPreparer → HybridRetriever → RAGGenerator
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ..data.prepare_dataset import DatasetPreparer
from .retriever import HybridRetriever
from .generator import RAGGenerator
from ..utils.logging_utils import get_logger
from ..utils.metrics import compute_all_metrics, print_metrics_table

logger = get_logger("RAGPipeline")


class RAGPipeline:
    """
    Full RAG pipeline: index → retrieve → generate → evaluate.
    """

    def __init__(self, config: Dict):
        self.config   = config
        self.retriever = HybridRetriever(config)
        self.generator: Optional[RAGGenerator] = None  # Lazy-load (heavy)

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

    def build_or_load_index(self, chunks: List[str], index_dir: str = "outputs/rag/index") -> None:
        """
        Build a new index or load an existing one from disk.

        Args:
            chunks:    Document chunks (only used when building fresh).
            index_dir: Directory to save/load index.
        """
        index_path = Path(index_dir)
        if (index_path / "faiss.index").exists():
            logger.info(f"Loading existing index from {index_dir}")
            self.retriever.load_index(index_dir)
        else:
            logger.info("Building new index...")
            self.retriever.build_index(chunks)
            self.retriever.save_index(index_dir)

    def _load_generator(self) -> None:
        """Lazy-load the generation model (only when needed for generation)."""
        if self.generator is None:
            self.generator = RAGGenerator(self.config)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, question: str, with_generation: bool = True) -> Dict:
        """
        Answer a single question using RAG.

        Args:
            question:        User's natural language question.
            with_generation: If False, return only retrieved chunks (faster).

        Returns:
            Dict with 'question', 'context', 'answer', 'search_time'.
        """
        chunks, scores, search_time = self.retriever.retrieve(question)
        result = {
            "question":    question,
            "context":     chunks,
            "scores":      scores,
            "search_time": search_time,
            "answer":      None,
        }
        if with_generation:
            self._load_generator()
            result["answer"] = self.generator.generate(question, chunks)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_queries_path: str,
        output_path: str = "outputs/rag/results.json",
        with_generation: bool = True,
        use_bertscore: bool   = False,
    ) -> Dict:
        """
        Run evaluation over a set of test queries.

        Args:
            test_queries_path: Path to JSON file with {question, answer} pairs.
            output_path:       Where to save per-query results.
            with_generation:   Whether to generate answers (or just retrieve).
            use_bertscore:     Compute BERTScore (slow).

        Returns:
            Aggregated metrics dict.
        """
        with open(test_queries_path, "r") as f:
            test_data = json.load(f)

        logger.info(f"Evaluating on {len(test_data)} test queries...")

        predictions = []
        references  = []
        all_results = []

        for item in test_data:
            question  = item["question"]
            reference = item["answer"]

            result = self.query(question, with_generation=with_generation)

            if with_generation and result["answer"]:
                prediction = result["answer"]
            else:
                # Use concatenated retrieved context as prediction
                prediction = " ".join(result["context"])

            predictions.append(prediction)
            references.append(reference)
            all_results.append({
                **result,
                "reference": reference,
                "prediction": prediction,
            })

            logger.info(f"Q: {question[:60]}...")
            logger.info(f"  search_time: {result['search_time']*1000:.2f}ms")

        # Compute metrics
        metrics = compute_all_metrics(predictions, references, use_bertscore=use_bertscore)
        print_metrics_table(metrics, model_name="DEEPHPC RAG")

        # Save results
        output = {
            "metrics":     metrics,
            "per_query":   all_results,
            "num_queries": len(test_data),
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        return metrics

    def run_grid_search(self, test_queries_path: str, output_path: str = "outputs/rag/grid_search.json") -> Dict:
        """
        Run FAISS hyperparameter grid search.
        """
        with open(test_queries_path, "r") as f:
            test_data = json.load(f)

        queries      = [d["question"] for d in test_data]
        ground_truth = [d["answer"] for d in test_data]

        gs_cfg     = self.config.get("grid_search", {})
        nlist_vals = gs_cfg.get("nlist_values", [1, 5, 10, 15, 20, 25])
        nprobe_vals= gs_cfg.get("nprobe_values", [1, 5, 10, 15, 20, 25])
        lambda_acc = gs_cfg.get("lambda_accuracy", 0.7)

        results = self.retriever.grid_search(
            queries, ground_truth, nlist_vals, nprobe_vals, lambda_acc
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Grid search results saved to {output_path}")
        return results
