"""
Evaluation metrics for DEEPHPC.

Improvements over original:
- ROUGE-L (measures longest common subsequence overlap)
- Cosine similarity via TF-IDF (matching original paper)
- BERTScore (deep semantic similarity — optional, GPU-heavy)
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from rouge_score import rouge_scorer


def compute_cosine_similarity(predictions: List[str], references: List[str]) -> Dict:
    """
    Compute TF-IDF cosine similarity between predictions and references.
    Matches the metric used in the original DEEPHPC paper.

    Args:
        predictions: List of model-generated answers.
        references:  List of ground-truth answers.

    Returns:
        dict with 'mean', 'scores' list.
    """
    assert len(predictions) == len(references), "Length mismatch"

    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = predictions + references
    vectorizer.fit(all_texts)

    pred_vecs = vectorizer.transform(predictions).toarray()
    ref_vecs  = vectorizer.transform(references).toarray()

    scores = []
    for p, r in zip(pred_vecs, ref_vecs):
        # Handle zero vectors (empty strings)
        if np.linalg.norm(p) == 0 or np.linalg.norm(r) == 0:
            scores.append(0.0)
        else:
            sim = float(sk_cosine(p.reshape(1, -1), r.reshape(1, -1))[0][0])
            scores.append(sim)

    return {"mean": float(np.mean(scores)), "scores": scores}


def compute_rouge_l(predictions: List[str], references: List[str]) -> Dict:
    """
    Compute ROUGE-L F1 between predictions and references.

    ROUGE-L measures the longest common subsequence (LCS) overlap,
    which is better than cosine similarity for capturing answer structure.

    Args:
        predictions: List of model-generated answers.
        references:  List of ground-truth answers.

    Returns:
        dict with 'mean', 'scores' list.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)

    return {"mean": float(np.mean(scores)), "scores": scores}


def compute_rouge1(predictions: List[str], references: List[str]) -> Dict:
    """Compute ROUGE-1 unigram F1."""
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = [scorer.score(r, p)["rouge1"].fmeasure for p, r in zip(predictions, references)]
    return {"mean": float(np.mean(scores)), "scores": scores}


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    use_bertscore: bool = False,
    device: str = "cpu",
) -> Dict:
    """
    Compute all evaluation metrics.

    Args:
        predictions:    Model-generated answers.
        references:     Ground-truth answers.
        use_bertscore:  If True, also compute BERTScore (slow, needs GPU).
        device:         Device for BERTScore ('cpu' or 'cuda').

    Returns:
        Dict with all metric results.
    """
    results = {}

    results["cosine_similarity"] = compute_cosine_similarity(predictions, references)
    results["rouge_l"]           = compute_rouge_l(predictions, references)
    results["rouge_1"]           = compute_rouge1(predictions, references)

    if use_bertscore:
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(predictions, references, lang="en", device=device, verbose=False)
            bs_scores = F1.tolist()
            results["bert_score"] = {"mean": float(np.mean(bs_scores)), "scores": bs_scores}
        except ImportError:
            print("bert-score not installed, skipping BERTScore.")

    return results


def print_metrics_table(results: Dict, model_name: str = "Model") -> None:
    """Pretty-print a metrics comparison table."""
    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title=f"Evaluation Results — {model_name}", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Score", style="green")

        metric_labels = {
            "cosine_similarity": "Cosine Similarity (TF-IDF)",
            "rouge_l":           "ROUGE-L F1",
            "rouge_1":           "ROUGE-1 F1",
            "bert_score":        "BERTScore F1",
        }

        for key, label in metric_labels.items():
            if key in results:
                table.add_row(label, f"{results[key]['mean']:.4f}")

        console.print(table)
    except ImportError:
        for key, val in results.items():
            print(f"  {key}: {val['mean']:.4f}")
