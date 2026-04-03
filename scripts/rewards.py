"""Reward functions for GRPO training.

Wraps the competition judger into reward functions compatible with TRL's
GRPOTrainer. Provides correctness-based and format-based rewards.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from judger import Judger

_judger = Judger(strict_extract=False)


def extract_boxed_content(text: str) -> str:
    """Extract the content inside the last \\boxed{} in model output."""
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    idx = search_text.rfind("\\boxed{")
    if idx < 0:
        return ""
    brace_start = idx + len("\\boxed{")
    depth, i = 1, brace_start
    while i < len(search_text) and depth > 0:
        if search_text[i] == '{':
            depth += 1
        elif search_text[i] == '}':
            depth -= 1
        i += 1
    return search_text[brace_start:i - 1].strip() if depth == 0 else ""


def correctness_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
    """Score each completion for mathematical correctness.

    Args:
        completions: List of conversations (each is a list of message dicts).
            The last message in each conversation is the assistant response.
        **kwargs: Must include ``answer`` (list of gold answers, one per sample).

    Returns:
        List of reward floats (1.0 for correct, 0.0 for incorrect).
    """
    gold_answers = kwargs.get("answer", [])
    rewards: list[float] = []

    for i, conversation in enumerate(completions):
        response = conversation[-1]["content"] if conversation else ""
        gold = gold_answers[i] if i < len(gold_answers) else None

        if gold is None:
            rewards.append(0.0)
            continue

        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            is_correct = _judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
            rewards.append(1.0 if is_correct else 0.0)
        except Exception:
            rewards.append(0.0)

    return rewards


def format_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
    """Reward proper formatting with \\boxed{} in the response.

    Returns 1.0 if \\boxed{} is present with non-empty content, 0.0 otherwise.
    """
    rewards: list[float] = []
    for conversation in completions:
        response = conversation[-1]["content"] if conversation else ""
        boxed = extract_boxed_content(response)
        rewards.append(1.0 if boxed else 0.0)
    return rewards


def combined_reward(
    completions: list[list[dict[str, str]]],
    correctness_weight: float = 1.0,
    format_weight: float = 0.1,
    **kwargs: Any,
) -> list[float]:
    """Weighted combination of correctness and format rewards."""
    c_rewards = correctness_reward(completions, **kwargs)
    f_rewards = format_reward(completions, **kwargs)
    return [
        correctness_weight * c + format_weight * f
        for c, f in zip(c_rewards, f_rewards)
    ]
