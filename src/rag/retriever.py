"""
Hybrid Retriever for DEEPHPC RAG.

Key improvement over original: combines FAISS (dense) + BM25 (sparse) retrieval
using Reciprocal Rank Fusion (RRF) for better recall across different query types.

- Dense retrieval: good for semantic/paraphrase queries
- Sparse (BM25): good for exact keyword/terminology queries (e.g., "SLURM", "sbatch")
- Hybrid: best of both worlds
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .embedder import DocumentEmbedder
from ..utils.logging_utils import get_logger

logger = get_logger("Retriever")


class HybridRetriever:
    """
    Hybrid BM25 + FAISS retriever with Reciprocal Rank Fusion.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Full RAG config dictionary.
        """
        self.config      = config
        faiss_cfg        = config.get("faiss", {})
        self.nlist       = faiss_cfg.get("nlist", 10)
        self.nprobe      = faiss_cfg.get("nprobe", 5)

        retrieval_cfg    = config.get("retrieval", {})
        self.mode        = retrieval_cfg.get("mode", "hybrid")   # dense | sparse | hybrid
        self.top_k       = retrieval_cfg.get("top_k", 5)
        self.dense_w     = retrieval_cfg.get("dense_weight", 0.6)
        self.sparse_w    = retrieval_cfg.get("sparse_weight", 0.4)

        emb_cfg          = config.get("embedding", {})
        self.embedder    = DocumentEmbedder(
            model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=emb_cfg.get("batch_size", 64),
            normalize=emb_cfg.get("normalize", True),
        )

        self.chunks: List[str]       = []
        self.embeddings: np.ndarray  = None
        self.faiss_index: faiss.Index = None
        self.bm25: BM25Okapi          = None

    # ─────────────────────────────────────────────────────────────────────────
    # Index building
    # ─────────────────────────────────────────────────────────────────────────

    def build_index(self, chunks: List[str]) -> None:
        """
        Build both FAISS and BM25 indexes from document chunks.

        Args:
            chunks: List of text chunks to index.
        """
        self.chunks = chunks
        logger.info(f"Building index for {len(chunks)} chunks...")

        # Dense index
        self.embeddings = self.embedder.encode_documents(chunks)
        self._build_faiss_index(self.embeddings)

        # Sparse index
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        logger.info("Index built successfully.")

    def _build_faiss_index(self, embeddings: np.ndarray, nlist: int = None, nprobe: int = None) -> None:
        """Build an IVF FAISS index from embeddings."""
        if nlist is None:
            nlist = self.nlist
        if nprobe is None:
            nprobe = self.nprobe

        dim = embeddings.shape[1]
        n   = embeddings.shape[0]

        # IVF requires at least nlist training points
        actual_nlist = min(nlist, max(1, n // 4))

        quantizer = faiss.IndexFlatIP(dim)   # Inner product (cosine for normalized vecs)
        index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe

        self.faiss_index = index
        logger.debug(f"FAISS IVF index: nlist={actual_nlist}, nprobe={nprobe}, n={n}")

    def save_index(self, save_dir: str) -> None:
        """Persist the index and chunks to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(save_path / "faiss.index"))
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(save_path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        np.save(str(save_path / "embeddings.npy"), self.embeddings)
        logger.info(f"Index saved to {save_dir}")

    def load_index(self, save_dir: str) -> None:
        """Load a previously saved index from disk."""
        save_path = Path(save_dir)

        self.faiss_index = faiss.read_index(str(save_path / "faiss.index"))
        self.faiss_index.nprobe = self.nprobe

        with open(save_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        with open(save_path / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        embeddings_path = save_path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(str(embeddings_path))

        logger.info(f"Index loaded from {save_dir} ({len(self.chunks)} chunks)")

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[str], List[float], float]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query:  User question string.
            top_k:  Override top_k from config.

        Returns:
            (chunks, scores, search_time_seconds)
        """
        if top_k is None:
            top_k = self.top_k

        t0 = time.perf_counter()

        if self.mode == "dense":
            chunks, scores = self._dense_retrieve(query, top_k)
        elif self.mode == "sparse":
            chunks, scores = self._sparse_retrieve(query, top_k)
        else:  # hybrid
            chunks, scores = self._hybrid_retrieve(query, top_k)

        elapsed = time.perf_counter() - t0
        return chunks, scores, elapsed

    def _dense_retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[float]]:
        """FAISS dense retrieval."""
        q_vec = self.embedder.encode_query(query)
        scores, indices = self.faiss_index.search(q_vec, top_k)
        retrieved = [(self.chunks[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        chunks, sc = zip(*retrieved) if retrieved else ([], [])
        return list(chunks), list(sc)

    def _sparse_retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[float]]:
        """BM25 sparse retrieval."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        chunks = [self.chunks[i] for i in top_indices]
        sc     = [float(scores[i]) for i in top_indices]
        return chunks, sc

    def _hybrid_retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[float]]:
        """
        Reciprocal Rank Fusion of dense + sparse results.

        RRF score for document d: sum(1 / (k + rank_i(d))) for each retrieval system i
        where k=60 is a smoothing constant.
        """
        k_rrf = 60
        # Get more candidates from each system, then fuse
        candidates = min(top_k * 3, len(self.chunks))

        # Dense candidates
        q_vec = self.embedder.encode_query(query)
        d_scores, d_indices = self.faiss_index.search(q_vec, candidates)
        dense_ranks = {
            int(idx): rank
            for rank, (idx, score) in enumerate(zip(d_indices[0], d_scores[0]))
            if idx >= 0
        }

        # Sparse candidates
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        sparse_top  = np.argsort(bm25_scores)[::-1][:candidates]
        sparse_ranks = {int(idx): rank for rank, idx in enumerate(sparse_top)}

        # RRF fusion
        all_indices = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        rrf_scores: Dict[int, float] = {}
        for idx in all_indices:
            score = 0.0
            if idx in dense_ranks:
                score += self.dense_w * (1.0 / (k_rrf + dense_ranks[idx]))
            if idx in sparse_ranks:
                score += self.sparse_w * (1.0 / (k_rrf + sparse_ranks[idx]))
            rrf_scores[idx] = score

        # Sort and return top_k
        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        chunks = [self.chunks[i] for i in sorted_indices]
        scores = [rrf_scores[i] for i in sorted_indices]
        return chunks, scores

    # ─────────────────────────────────────────────────────────────────────────
    # FAISS hyperparameter grid search
    # ─────────────────────────────────────────────────────────────────────────

    def grid_search(
        self,
        queries: List[str],
        ground_truth: List[str],
        nlist_values: List[int],
        nprobe_values: List[int],
        lambda_acc: float = 0.7,
    ) -> Dict:
        """
        Grid search over FAISS (nlist, nprobe) using a fitness function:
            F = lambda * cosine_sim - (1 - lambda) * search_time

        Args:
            queries:       List of evaluation queries.
            ground_truth:  List of reference answers.
            nlist_values:  Values of nlist to test.
            nprobe_values: Values of nprobe to test.
            lambda_acc:    Weight for accuracy in fitness (0-1).

        Returns:
            Dict with all results and best config.
        """
        from ..utils.metrics import compute_cosine_similarity

        results = []
        best_fitness = -np.inf
        best_config  = None

        total = len(nlist_values) * len(nprobe_values)
        done  = 0

        for nlist in nlist_values:
            # Rebuild FAISS index with this nlist
            self._build_faiss_index(self.embeddings, nlist=nlist, nprobe=1)

            for nprobe in nprobe_values:
                self.faiss_index.nprobe = nprobe
                done += 1
                logger.info(f"Grid search [{done}/{total}]: nlist={nlist}, nprobe={nprobe}")

                # Retrieve and measure time
                retrieved_chunks = []
                search_times     = []
                for query in queries:
                    chunks, _, elapsed = self.retrieve(query, top_k=5)
                    retrieved_chunks.append(" ".join(chunks))
                    search_times.append(elapsed)

                avg_time = float(np.mean(search_times))

                # Accuracy: cosine similarity between retrieved context and ground truth
                cos = compute_cosine_similarity(retrieved_chunks, ground_truth)
                avg_acc = cos["mean"]

                # Fitness function (from paper, eq. 2)
                fitness = lambda_acc * avg_acc - (1 - lambda_acc) * avg_time

                row = {
                    "nlist":    nlist,
                    "nprobe":   nprobe,
                    "accuracy": avg_acc,
                    "avg_time": avg_time,
                    "fitness":  fitness,
                }
                results.append(row)
                logger.info(f"  acc={avg_acc:.4f}, time={avg_time:.6f}s, fitness={fitness:.4f}")

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_config  = (nlist, nprobe)

        # Restore best config
        best_nlist, best_nprobe = best_config
        self._build_faiss_index(self.embeddings, nlist=best_nlist, nprobe=best_nprobe)
        logger.info(f"Best config: nlist={best_nlist}, nprobe={best_nprobe}, fitness={best_fitness:.4f}")

        return {
            "results":     results,
            "best_nlist":  best_nlist,
            "best_nprobe": best_nprobe,
            "best_fitness": best_fitness,
        }
