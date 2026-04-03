"""QLoRA Supervised Fine-Tuning for Qwen3-4B on math reasoning data.

Fine-tunes the model using TRL's SFTTrainer with 4-bit quantization and LoRA
adapters. Designed to run on A100/H100 GPUs on a cluster.

Usage:
    python scripts/train_sft.py
    python scripts/train_sft.py --config configs/sft_config.yaml
    accelerate launch scripts/train_sft.py --config configs/sft_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def load_training_data(data_path: str) -> Dataset:
    """Load JSONL training data as a HuggingFace Dataset.

    Each line must have a ``messages`` field containing a list of chat messages.
    """
    records: list[dict[str, Any]] = []
    full_path = PROJECT_ROOT / data_path
    with open(full_path) as f:
        for line in f:
            item = json.loads(line.strip())
            if "messages" in item:
                records.append(item)

    logger.info("Loaded %d training examples from %s", len(records), full_path)
    return Dataset.from_list(records)


def build_quantization_config(quant_cfg: dict[str, Any]) -> BitsAndBytesConfig:
    """Build BitsAndBytes quantization config from YAML settings."""
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype_map.get(
            quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16
        ),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_lora_config(lora_cfg: dict[str, Any]) -> LoraConfig:
    """Build PEFT LoRA config from YAML settings."""
    return LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        bias="none",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="QLoRA SFT training for math reasoning")
    parser.add_argument(
        "--config", default="configs/sft_config.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """Run QLoRA SFT training."""
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    model_name = model_cfg["name_or_path"]
    output_dir = str(PROJECT_ROOT / train_cfg["output_dir"])

    # Load tokenizer
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    logger.info("Loading model with 4-bit quantization ...")
    bnb_config = build_quantization_config(quant_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Apply LoRA
    peft_config = build_lora_config(lora_cfg)
    logger.info("LoRA config: r=%d, alpha=%d, targets=%s",
                peft_config.r, peft_config.lora_alpha, peft_config.target_modules)

    # If resuming from a previous adapter, load it
    adapter_path = model_cfg.get("adapter_path")
    if adapter_path:
        from peft import PeftModel
        logger.info("Loading existing adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # Load data
    train_dataset = load_training_data(data_cfg["train_path"])

    # Build SFT training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        max_seq_length=train_cfg.get("max_seq_length", 4096),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        report_to=train_cfg.get("report_to", "none"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training ...")
    trainer.train()

    # Save final adapter
    final_path = Path(output_dir) / "final"
    logger.info("Saving final adapter to %s", final_path)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
