"""
DEEPHPC Fine-Tuning with QLoRA.

Key improvements over original:
1. QLoRA (4-bit NF4 quantization) — ~4x less VRAM than plain LoRA
2. Target MLP layers in addition to attention (gate_proj, up_proj, down_proj)
3. Cosine LR schedule + warmup
4. Paged AdamW 8-bit optimizer (better memory for large batches)
5. Proper validation split evaluation each epoch
6. Training loss curve saved to JSON for plotting
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from .dataset import DeepHPCDataset
from ..utils.logging_utils import get_logger

logger = get_logger("FineTuner")


class LossLoggerCallback(TrainerCallback):
    """Callback to capture per-step losses for plotting."""

    def __init__(self):
        self.history = {"steps": [], "train_loss": [], "eval_loss": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if "loss" in logs:
                self.history["steps"].append(step)
                self.history["train_loss"].append(logs["loss"])
            if "eval_loss" in logs:
                self.history["eval_loss"].append({"step": step, "loss": logs["eval_loss"]})


class FineTuner:
    """
    QLoRA-based fine-tuner for DeepSeek-R1-Distill-Qwen-1.5B on ULHPC docs.
    """

    def __init__(self, config: Dict):
        self.config   = config
        self.model_cfg= config["model"]
        self.lora_cfg = config["lora"]
        self.qlora_cfg= config.get("qlora", {})
        self.train_cfg= config["training"]

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model_and_tokenizer(self):
        """Load base model with QLoRA quantization and apply LoRA adapters."""
        model_name = self.model_cfg["name"]
        cache_dir  = self.model_cfg.get("cache_dir", None)

        logger.info(f"Loading base model: {model_name}")

        # QLoRA 4-bit quantization config
        use_4bit = self.qlora_cfg.get("use_4bit", True)
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.qlora_cfg.get("use_nested_quant", True),
            )
            logger.info("Using QLoRA (4-bit NF4 quantization)")
        else:
            bnb_config = None
            logger.info("Using plain LoRA (no quantization)")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.pad_token     = tokenizer.eos_token
        tokenizer.padding_side  = "right"

        # Prepare model for k-bit training (required for QLoRA)
        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA adapters
        lora_config = LoraConfig(
            r              = self.lora_cfg["r"],
            lora_alpha     = self.lora_cfg["lora_alpha"],
            target_modules = self.lora_cfg["target_modules"],
            lora_dropout   = self.lora_cfg["lora_dropout"],
            bias           = self.lora_cfg["bias"],
            task_type      = TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        # Required when using gradient_checkpointing with LoRA on a frozen base model.
        # Without this, gradient flow is broken because frozen embedding inputs have
        # requires_grad=False, which causes backward() to fail.
        model.enable_input_require_grads()
        model.print_trainable_parameters()

        return model, tokenizer

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, train_path: str, val_path: Optional[str] = None) -> str:
        """
        Run the full fine-tuning pipeline.

        Args:
            train_path: Path to training Q&A JSON.
            val_path:   Path to validation Q&A JSON (optional).

        Returns:
            Path to saved LoRA adapter directory.
        """
        # Build datasets
        dataset_builder = DeepHPCDataset(self.config)
        train_ds, val_ds = dataset_builder.build(train_path, val_path)

        # Load model + tokenizer
        model, tokenizer = self._load_model_and_tokenizer()

        # Training arguments
        output_dir = self.train_cfg["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir                  = output_dir,
            per_device_train_batch_size = self.train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size  = self.train_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps = self.train_cfg["gradient_accumulation_steps"],
            num_train_epochs            = self.train_cfg["num_train_epochs"],
            learning_rate               = float(self.train_cfg["learning_rate"]),
            lr_scheduler_type           = self.train_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio                = self.train_cfg.get("warmup_ratio", 0.03),
            max_grad_norm               = self.train_cfg.get("max_grad_norm", 0.3),
            weight_decay                = self.train_cfg.get("weight_decay", 0.001),
            fp16                        = self.train_cfg.get("fp16", True),
            optim                       = self.train_cfg.get("optim", "adamw_torch"),
            save_strategy               = self.train_cfg.get("save_strategy", "epoch"),
            save_steps                  = self.train_cfg.get("save_steps", 100),
            logging_steps               = self.train_cfg.get("logging_steps", 25),
            eval_strategy               = self.train_cfg.get("eval_strategy", "epoch") if val_ds else "no",
            save_total_limit            = self.train_cfg.get("save_total_limit", 2),
            load_best_model_at_end      = bool(val_ds),
            metric_for_best_model       = "eval_loss" if val_ds else None,
            report_to                   = "none",
            dataloader_num_workers      = 0,
            gradient_checkpointing      = True,   # saves ~40% VRAM on 16GB nodes
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        loss_logger   = LossLoggerCallback()

        trainer = Trainer(
            model         = model,
            args          = training_args,
            train_dataset = train_ds,
            eval_dataset  = val_ds,
            tokenizer     = tokenizer,
            data_collator = data_collator,
            callbacks     = [loss_logger],
        )

        logger.info("Starting training...")
        trainer.train()

        # Save final adapter
        adapter_path = self.config["inference"]["adapter_path"]
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        logger.info(f"LoRA adapter saved to: {adapter_path}")

        # Save loss history for plotting
        loss_path = Path(self.train_cfg["output_dir"]) / "loss_history.json"
        with open(loss_path, "w") as f:
            json.dump(loss_logger.history, f, indent=2)
        logger.info(f"Loss history saved to: {loss_path}")

        return adapter_path
