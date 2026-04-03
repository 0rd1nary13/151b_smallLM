"""Prepare and format math training data for SFT.

Downloads, filters, and formats math reasoning datasets into the Qwen3 chat
template with <think>...</think> reasoning blocks. Outputs a single JSONL
training file ready for TRL's SFTTrainer.

Datasets used:
  - AI-MO/NuminaMath-CoT  (competition math with chain-of-thought)
  - meta-math/MetaMathQA   (augmented GSM8K + MATH)

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --max-samples 50000
    python scripts/prepare_data.py --output data/train_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def format_numina_example(example: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a NuminaMath-CoT example into chat messages.

    Returns None if the example should be skipped (e.g., missing fields).
    """
    question = example.get("problem", "").strip()
    solution = example.get("solution", "").strip()

    if not question or not solution:
        return None

    # Skip extremely short solutions (likely low quality)
    if len(solution) < 50:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": solution},
    ]
    return {"messages": messages}


def format_metamath_example(example: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a MetaMathQA example into chat messages.

    Returns None if the example should be skipped.
    """
    question = example.get("query", "").strip()
    response = example.get("response", "").strip()

    if not question or not response:
        return None

    if len(response) < 50:
        return None

    # MetaMathQA responses often end with "The answer is: X"
    # Wrap the final answer in \boxed{} if not already present
    if "\\boxed{" not in response and "The answer is:" in response:
        parts = response.rsplit("The answer is:", 1)
        if len(parts) == 2:
            answer_val = parts[1].strip().rstrip(".")
            response = parts[0] + f"The answer is $\\boxed{{{answer_val}}}$."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    return {"messages": messages}


def load_numina(max_samples: int | None) -> list[dict[str, Any]]:
    """Load and format NuminaMath-CoT dataset."""
    print("Loading AI-MO/NuminaMath-CoT ...")
    try:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        print(f"  Warning: could not load NuminaMath-CoT: {e}")
        return []

    formatted: list[dict[str, Any]] = []
    for example in tqdm(ds, desc="  Formatting NuminaMath"):
        result = format_numina_example(example)
        if result is not None:
            formatted.append(result)

    print(f"  NuminaMath: {len(formatted)} examples after filtering")

    if max_samples and len(formatted) > max_samples:
        random.shuffle(formatted)
        formatted = formatted[:max_samples]
        print(f"  Sampled down to {len(formatted)}")

    return formatted


def load_metamath(max_samples: int | None) -> list[dict[str, Any]]:
    """Load and format MetaMathQA dataset."""
    print("Loading meta-math/MetaMathQA ...")
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train")
    except Exception as e:
        print(f"  Warning: could not load MetaMathQA: {e}")
        return []

    formatted: list[dict[str, Any]] = []
    for example in tqdm(ds, desc="  Formatting MetaMathQA"):
        result = format_metamath_example(example)
        if result is not None:
            formatted.append(result)

    print(f"  MetaMathQA: {len(formatted)} examples after filtering")

    if max_samples and len(formatted) > max_samples:
        random.shuffle(formatted)
        formatted = formatted[:max_samples]
        print(f"  Sampled down to {len(formatted)}")

    return formatted


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare math SFT training data")
    parser.add_argument(
        "--output", default="data/train_sft.jsonl",
        help="Output JSONL path (relative to project root)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=80000,
        help="Max total samples in the final dataset (None = all)",
    )
    parser.add_argument(
        "--numina-fraction", type=float, default=0.6,
        help="Fraction of samples from NuminaMath (rest from MetaMathQA)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Prepare and save the SFT training dataset."""
    args = parse_args()
    random.seed(args.seed)

    numina_max = int(args.max_samples * args.numina_fraction) if args.max_samples else None
    meta_max = int(args.max_samples * (1 - args.numina_fraction)) if args.max_samples else None

    numina_data = load_numina(numina_max)
    meta_data = load_metamath(meta_max)

    all_data = numina_data + meta_data
    random.shuffle(all_data)

    if args.max_samples and len(all_data) > args.max_samples:
        all_data = all_data[:args.max_samples]

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(all_data)} training examples to {out_path}")
    print(f"  NuminaMath: {len(numina_data)}")
    print(f"  MetaMathQA: {len(meta_data)}")


if __name__ == "__main__":
    main()
