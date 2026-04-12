"""
DEEPHPC Data Preparation Script.

Usage:
    python scripts/prepare_data.py --config configs/finetune_config.yaml \
                                   --rag-config configs/rag_config.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prepare_dataset import DatasetPreparer
from src.utils.logging_utils import get_logger

logger = get_logger("PrepareData")


def main():
    parser = argparse.ArgumentParser(description="Prepare DEEPHPC dataset")
    parser.add_argument("--config",     required=True, help="Fine-tune config YAML")
    parser.add_argument("--rag-config", required=True, help="RAG config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        ft_config = yaml.safe_load(f)
    with open(args.rag_config) as f:
        rag_config = yaml.safe_load(f)

    # Merge configs for DatasetPreparer (it uses keys from both)
    merged = {**rag_config, **ft_config}

    preparer = DatasetPreparer(merged)

    # Clone ULHPC docs
    repo_url   = rag_config["docs"]["repo_url"]
    local_path = rag_config["docs"]["local_path"]
    preparer.clone_docs(repo_url, local_path)

    # Build Q&A dataset
    pairs = preparer.build_qa_dataset()

    # Save splits
    train_path, val_path = preparer.save_dataset(pairs)

    logger.info(f"Dataset ready: {len(pairs)} total Q&A pairs")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Val:   {val_path}")


if __name__ == "__main__":
    main()
