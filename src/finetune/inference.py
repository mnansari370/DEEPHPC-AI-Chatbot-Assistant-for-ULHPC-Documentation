"""
Inference with fine-tuned DEEPHPC model (LoRA/QLoRA adapters).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..utils.logging_utils import get_logger
from ..utils.metrics import compute_all_metrics, print_metrics_table

logger = get_logger("FTInference")

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


class FineTunedModel:
    """
    Loads a fine-tuned DeepSeek model (with LoRA adapters) for inference.
    """

    def __init__(self, config: Dict):
        self.config     = config
        infer_cfg       = config.get("inference", {})
        self.adapter_path = infer_cfg.get("adapter_path", "outputs/finetune/final_adapter")
        self.max_tokens   = infer_cfg.get("max_new_tokens", 300)
        self.temperature  = infer_cfg.get("temperature", 0.1)
        self.do_sample    = infer_cfg.get("do_sample", False)
        self.model_name   = config["model"]["name"]
        self._loaded      = False

    def _load(self) -> None:
        """Lazy-load the model (avoids loading on import)."""
        if self._loaded:
            return

        logger.info(f"Loading base model: {self.model_name}")
        base = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self._loaded = True
        logger.info("Fine-tuned model ready.")

    def answer(self, question: str) -> str:
        """
        Generate an answer to a question using the fine-tuned model.

        Args:
            question: User's natural language question.

        Returns:
            Generated answer string.
        """
        self._load()
        prompt = PROMPT_TEMPLATE.format(instruction=question)
        output = self.pipe(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0]["generated_text"]
        return generated[len(prompt):].strip()

    def evaluate(
        self,
        test_queries_path: str,
        output_path: str = "outputs/finetune/results.json",
        use_bertscore: bool = False,
    ) -> Dict:
        """
        Evaluate the fine-tuned model on test queries.

        Args:
            test_queries_path: Path to JSON with {question, answer} pairs.
            output_path:       Where to save results.
            use_bertscore:     Also compute BERTScore.

        Returns:
            Aggregated metrics dict.
        """
        self._load()

        with open(test_queries_path, "r") as f:
            test_data = json.load(f)

        logger.info(f"Evaluating fine-tuned model on {len(test_data)} queries...")

        predictions = []
        references  = []
        per_query   = []

        for item in test_data:
            question  = item["question"]
            reference = item["answer"]

            prediction = self.answer(question)
            predictions.append(prediction)
            references.append(reference)

            per_query.append({
                "question":   question,
                "prediction": prediction,
                "reference":  reference,
            })
            logger.info(f"Q: {question[:60]}...")
            logger.info(f"A: {prediction[:80]}...")

        metrics = compute_all_metrics(predictions, references, use_bertscore=use_bertscore)
        print_metrics_table(metrics, model_name="DEEPHPC Fine-Tuned")

        output = {
            "metrics":     metrics,
            "per_query":   per_query,
            "num_queries": len(test_data),
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Evaluation saved to {output_path}")

        return metrics
