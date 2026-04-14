"""
Response Generator for DEEPHPC RAG.

Loads DeepSeek-R1-Distill-Qwen-1.5B and generates answers given
a user query + retrieved context chunks.
"""
from __future__ import annotations

from typing import List, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ..utils.logging_utils import get_logger

logger = get_logger("Generator")

SYSTEM_PROMPT = """You are DEEPHPC, an expert assistant for the University of Luxembourg \
High Performance Computing (ULHPC) cluster. Answer questions clearly and concisely based \
on the provided documentation context. If the context does not contain enough information, \
say so honestly rather than guessing."""


class RAGGenerator:
    """
    Wraps a causal LM for RAG-style generation.
    Combines user query + retrieved chunks into a structured prompt.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Full RAG config (uses config["generation"]).
        """
        gen_cfg         = config.get("generation", {})
        self.model_name = gen_cfg.get("model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.max_tokens = gen_cfg.get("max_new_tokens", 300)
        self.temperature= gen_cfg.get("temperature", 0.1)
        self.do_sample  = gen_cfg.get("do_sample", False)
        self.device     = gen_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading generator model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        # Do NOT pass `device` when the model was loaded with device_map="auto"
        # (HuggingFace raises ValueError if both device and device_map are set).
        # The pipeline auto-detects placement from the already-loaded model.
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info("Generator ready.")

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate an answer given a query and retrieved context.

        Args:
            query:          User's natural language question.
            context_chunks: Top-k retrieved document chunks.

        Returns:
            Generated answer string.
        """
        context = "\n\n---\n\n".join(context_chunks)
        prompt  = self._build_prompt(query, context)

        output = self.pipe(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0]["generated_text"]

        # Strip the prompt prefix — return only the new tokens
        answer = generated[len(prompt):].strip()
        return answer

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build a structured prompt for the RAG pipeline.
        Uses DeepSeek instruction format.
        """
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"### Context from ULHPC Documentation:\n{context}\n\n"
            f"### Question:\n{query}\n\n"
            f"### Answer:\n"
        )
