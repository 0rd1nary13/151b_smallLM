"""GRPO Reinforcement Learning training for Qwen3-4B.

Applies Group Relative Policy Optimization starting from the SFT checkpoint,
using math correctness as the reward signal.

Usage:
    python scripts/train_grpo.py
    python scripts/train_grpo.py --config configs/grpo_config.yaml
    accelerate launch scripts/train_grpo.py --config configs/grpo_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM_PROMPT = (
    "You are an expert mathematician who solves problems with rigorous step-by-step reasoning.\n\n"
    "Instructions:\n"
    "1. Break the problem into clear steps. Show all work.\n"
    "2. For each step, state what you are doing and why.\n"
    "3. Double-check your arithmetic and algebra before concluding.\n"
    "4. Put your final answer inside \\boxed{}.\n"
    "5. If the problem asks for multiple sub-answers, separate them by commas "
    "inside a single \\boxed{}, e.g. \\boxed{3, 7}.\n"
    "6. Give exact answers (fractions, radicals, pi) unless a decimal approximation is explicitly requested.\n"
    "7. If the answer is a well-known constant or expression, simplify it fully."
)


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def load_grpo_dataset(data_path: str, max_samples: int | None = None) -> Dataset:
    """Load training data and format as prompt-only for GRPO.

    Each item needs a ``prompt`` field (list of messages) and an ``answer``
    field for the reward function.
    """
    records: list[dict[str, Any]] = []
    full_path = PROJECT_ROOT / data_path

    with open(full_path) as f:
        for line in f:
            item = json.loads(line.strip())
            if "messages" not in item:
                continue

            messages = item["messages"]
            # Extract prompt (system + user) and answer (assistant) for reward
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

            if not assistant_msg:
                continue

            # Extract the gold answer from the assistant response for reward evaluation
            answer_text = assistant_msg["content"]
            records.append({
                "prompt": prompt_messages,
                "answer": answer_text,
            })

    if max_samples and len(records) > max_samples:
        random.shuffle(records)
        records = records[:max_samples]

    logger.info("Loaded %d GRPO training examples from %s", len(records), full_path)
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
    parser = argparse.ArgumentParser(description="GRPO RL training for math reasoning")
    parser.add_argument(
        "--config", default="configs/grpo_config.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def make_correctness_reward_fn(judger: Any) -> callable:
    """Create a correctness reward function using the competition judger.

    The returned function extracts the gold answer from the dataset's
    ``answer`` column and checks the model's completion against it.
    """
    import re

    def _extract_boxed(text: str) -> str:
        think_end = text.rfind("</think>")
        s = text[think_end + len("</think>"):] if think_end >= 0 else text
        idx = s.rfind("\\boxed{")
        if idx < 0:
            return ""
        start = idx + len("\\boxed{")
        depth, i = 1, start
        while i < len(s) and depth > 0:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
            i += 1
        return s[start:i - 1].strip() if depth == 0 else ""

    def reward_fn(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
        """Score completions against gold answers."""
        gold_answers = kwargs.get("answer", [])
        rewards: list[float] = []

        for i, conversation in enumerate(completions):
            response = conversation[-1]["content"] if conversation else ""

            # Format reward component
            has_boxed = 1.0 if _extract_boxed(response) else 0.0

            # Correctness reward
            gold = gold_answers[i] if i < len(gold_answers) else None
            if gold is None:
                rewards.append(0.1 * has_boxed)
                continue

            # The gold here is the full assistant response from SFT data;
            # extract the boxed answer from it for comparison
            gold_boxed = _extract_boxed(gold)
            if gold_boxed:
                gold_list = [gold_boxed]
            else:
                gold_list = [gold] if not isinstance(gold, list) else gold

            try:
                is_correct = judger.auto_judge(
                    pred=response, gold=gold_list,
                    options=[[]] * len(gold_list),
                )
                rewards.append(1.0 if is_correct else 0.1 * has_boxed)
            except Exception:
                rewards.append(0.1 * has_boxed)

        return rewards

    return reward_fn


def main() -> None:
    """Run GRPO training."""
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    reward_cfg = cfg.get("rewards", {})

    model_name = model_cfg["name_or_path"]
    output_dir = str(PROJECT_ROOT / train_cfg["output_dir"])

    # Load tokenizer
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    logger.info("Loading model with 4-bit quantization ...")
    bnb_config = build_quantization_config(quant_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Load SFT adapter as starting point if specified
    sft_adapter = model_cfg.get("sft_adapter_path")
    if sft_adapter:
        from peft import PeftModel
        adapter_path = str(PROJECT_ROOT / sft_adapter)
        logger.info("Loading SFT adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info("Merged SFT adapter into base model")

    # LoRA config for GRPO training
    peft_config = build_lora_config(lora_cfg)

    # Load dataset
    train_dataset = load_grpo_dataset(
        data_cfg["train_path"],
        max_samples=data_cfg.get("max_samples"),
    )

    # Build reward function
    from judger import Judger
    judger = Judger(strict_extract=False)
    reward_fn = make_correctness_reward_fn(judger)

    # Build GRPO config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        max_completion_length=train_cfg.get("max_completion_length", 4096),
        max_prompt_length=train_cfg.get("max_prompt_length", 1024),
        num_generations=train_cfg.get("num_generations", 8),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 5),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "none"),
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting GRPO training ...")
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    logger.info("Saving final adapter to %s", final_path)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    logger.info("GRPO training complete!")


if __name__ == "__main__":
    main()
