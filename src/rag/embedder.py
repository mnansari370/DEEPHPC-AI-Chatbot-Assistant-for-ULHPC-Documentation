"""
Document Embedding Module for DEEPHPC RAG.

Uses SentenceTransformers all-MiniLM-L6-v2 to encode chunks and queries
into dense vectors for semantic similarity search.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from ..utils.logging_utils import get_logger

logger = get_logger("Embedder")


class DocumentEmbedder:
    """
    Wraps SentenceTransformers for batched encoding of document chunks and queries.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64, normalize: bool = True, device: str = "cpu"):
        """
        Args:
            model_name:  HuggingFace model identifier.
            batch_size:  Encoding batch size.
            normalize:   If True, L2-normalize embeddings (required for cosine similarity).
            device:      'cpu' or 'cuda'.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize
        self.device     = device

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of document chunks into dense vectors.

        Args:
            texts: List of text chunks.

        Returns:
            float32 array of shape (N, embedding_dim).
        """
        logger.info(f"Encoding {len(texts)} document chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Args:
            query: User question string.

        Returns:
            float32 array of shape (1, embedding_dim).
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)
