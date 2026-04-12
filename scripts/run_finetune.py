"""
DEEPHPC Fine-Tuning Script.

Modes:
  train    — Fine-tune DeepSeek-R1-Distill-Qwen-1.5B with QLoRA
  evaluate — Evaluate saved fine-tuned adapter on test queries
  query    — Answer a single question with the fine-tuned model

Usage:
    python scripts/run_finetune.py --config configs/finetune_config.yaml \
        --mode train --train-data data/qa_train.json --val-data data/qa_val.json

    python scripts/run_finetune.py --config configs/finetune_config.yaml \
        --mode evaluate --test-queries evaluation/test_queries.json

    python scripts/run_finetune.py --config configs/finetune_config.yaml \
        --mode query --question "How do I submit a GPU job on ULHPC?"
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finetune.train import FineTuner
from src.finetune.inference import FineTunedModel
from src.utils.logging_utils import get_logger

logger = get_logger("RunFinetune")


def main():
    parser = argparse.ArgumentParser(description="DEEPHPC Fine-Tuning Pipeline")
    parser.add_argument("--config",       required=True, help="Fine-tune config YAML")
    parser.add_argument("--mode",         required=True,
                        choices=["train", "evaluate", "query"])
    parser.add_argument("--train-data",   default="data/qa_train.json")
    parser.add_argument("--val-data",     default="data/qa_val.json")
    parser.add_argument("--test-queries", default="evaluation/test_queries.json")
    parser.add_argument("--output",       default=None)
    parser.add_argument("--question",     default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Train ────────────────────────────────────────────────────────────────
    if args.mode == "train":
        tuner = FineTuner(config)
        adapter_path = tuner.train(
            train_path=args.train_data,
            val_path=args.val_data if Path(args.val_data).exists() else None,
        )
        logger.info(f"Training complete. Adapter saved to: {adapter_path}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    elif args.mode == "evaluate":
        model  = FineTunedModel(config)
        output = args.output or config["evaluation"]["output_path"]
        metrics = model.evaluate(args.test_queries, output)
        logger.info(f"ROUGE-L: {metrics['rouge_l']['mean']:.4f}")
        logger.info(f"Cosine:  {metrics['cosine_similarity']['mean']:.4f}")

    # ── Interactive query ─────────────────────────────────────────────────────
    elif args.mode == "query":
        model    = FineTunedModel(config)
        question = args.question or input("Enter your question: ")
        answer   = model.answer(question)

        print("\n" + "="*60)
        print(f"QUESTION: {question}")
        print("-"*60)
        print(f"ANSWER:\n{answer}")
        print("="*60)


if __name__ == "__main__":
    main()
