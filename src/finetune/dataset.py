"""
Dataset and tokenization utilities for fine-tuning DEEPHPC.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from ..utils.logging_utils import get_logger

logger = get_logger("FTDataset")

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)


def load_qa_pairs(json_path: str) -> List[Dict]:
    """Load Q&A pairs from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} Q&A pairs from {json_path}")
    return pairs


def format_pair(example: Dict) -> str:
    """Format a single Q&A pair into instruction-following prompt."""
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        response=example["response"],
    )


class DeepHPCDataset:
    """
    Tokenized HuggingFace Dataset for fine-tuning DeepSeek with LoRA/QLoRA.
    """

    def __init__(self, config: Dict):
        tok_cfg          = config.get("tokenization", {})
        self.max_length  = tok_cfg.get("max_seq_length", 512)
        model_name       = config["model"]["name"]

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=config["model"].get("cache_dir", None),
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def build(self, train_path: str, val_path: Optional[str] = None):
        """
        Build train (and optional val) HuggingFace Datasets.

        Args:
            train_path: Path to JSON with training Q&A pairs.
            val_path:   Path to JSON with validation Q&A pairs (optional).

        Returns:
            (train_dataset, val_dataset) — val_dataset may be None.
        """
        train_pairs = load_qa_pairs(train_path)
        train_ds    = self._tokenize(train_pairs, split="train")

        val_ds = None
        if val_path and Path(val_path).exists():
            val_pairs = load_qa_pairs(val_path)
            val_ds    = self._tokenize(val_pairs, split="val")

        return train_ds, val_ds

    def _tokenize(self, pairs: List[Dict], split: str) -> HFDataset:
        """Tokenize a list of Q&A pairs."""
        texts = [format_pair(p) for p in pairs]

        def tokenize_fn(batch):
            tokenized = self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        raw_ds = HFDataset.from_dict({"text": texts})
        tokenized_ds = raw_ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {split} set",
        )
        logger.info(f"{split} dataset: {len(tokenized_ds)} examples")
        return tokenized_ds
