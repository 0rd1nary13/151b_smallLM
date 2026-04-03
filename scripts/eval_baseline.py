"""Standalone evaluation script for the math reasoning competition.

Runs inference with Qwen3-4B (optionally with a LoRA adapter) on the public
dataset and reports accuracy broken down by question type. Supports
self-consistency majority voting for both MCQ and free-form questions.

Usage:
    python scripts/eval_baseline.py                         # defaults
    python scripts/eval_baseline.py --subset 20             # quick smoke test
    python scripts/eval_baseline.py --adapter path/to/lora  # evaluate fine-tuned model
    python scripts/eval_baseline.py --mcq-n 8 --free-n 8   # self-consistency
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

# Ensure project root is on sys.path so judger.py is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT_MATH = (
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

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician who solves problems with rigorous step-by-step reasoning.\n\n"
    "Instructions:\n"
    "1. First, solve the problem yourself step-by-step WITHOUT looking at the options.\n"
    "2. Then compare your solution to each option to find the match.\n"
    "3. If your answer does not match any option, re-examine your work and "
    "try alternative approaches.\n"
    "4. Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list[str]]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_prompt = (
            f"{question}\n\n"
            f"Options:\n{opts_text}\n\n"
            f"Solve the problem step-by-step, then select the correct option letter."
        )
        return SYSTEM_PROMPT_MCQ, user_prompt
    user_prompt = (
        f"{question}\n\n"
        f"Solve this problem step-by-step, showing all work. "
        f"Put your final answer in \\boxed{{}}."
    )
    return SYSTEM_PROMPT_MATH, user_prompt


# ── Answer extraction & majority voting ───────────────────────────────────────

def extract_letter(text: str) -> str:
    """Extract a single letter from \\boxed{} or fallback to last capital letter."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def extract_boxed_answer(text: str) -> str:
    """Extract the raw content inside the last \\boxed{} after </think> tags."""
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    idx = search_text.rfind("\\boxed{")
    if idx < 0:
        return ""
    brace_start = idx + len("\\boxed{")
    depth = 1
    i = brace_start
    while i < len(search_text) and depth > 0:
        if search_text[i] == '{':
            depth += 1
        elif search_text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[brace_start:i - 1].strip()
    return ""


def pick_mcq_majority(outputs: list[Any]) -> str:
    """Majority vote on boxed letter across N completions."""
    outs = list(outputs)
    letters = [extract_letter(o.text.strip()) for o in outs]
    valid = [l for l in letters if l]
    if not valid:
        return outs[0].text.strip()
    winner, _ = Counter(valid).most_common(1)[0]
    for o, l in zip(outs, letters):
        if l == winner:
            return o.text.strip()
    return outs[0].text.strip()


def pick_freeform_majority(outputs: list[Any]) -> str:
    """Majority vote on boxed answer string across N completions."""
    outs = list(outputs)
    answers = [extract_boxed_answer(o.text.strip()) for o in outs]
    valid = [(ans, i) for i, ans in enumerate(answers) if ans]
    if not valid:
        return outs[0].text.strip()

    normalized = [re.sub(r"\s+", " ", a).strip().lower() for a, _ in valid]
    counter = Counter(normalized)
    winner_norm, _ = counter.most_common(1)[0]

    for (ans, idx), norm in zip(valid, normalized):
        if norm == winner_norm:
            return outs[idx].text.strip()
    return outs[0].text.strip()


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_mcq(response: str, gold_letter: str) -> bool:
    """Check if the extracted letter matches the gold answer."""
    return extract_letter(response) == gold_letter.strip().upper()


def score_freeform(judger: Any, response: str, gold: Any) -> bool:
    """Score a free-form response using the competition judger."""
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-4B on math competition dataset")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507", help="Base model ID")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--data", default="data/public.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output", default="results/eval_results.jsonl", help="Output path")
    parser.add_argument("--subset", type=int, default=None, help="Evaluate only first N questions")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--mcq-n", type=int, default=1, help="Self-consistency N for MCQ")
    parser.add_argument("--free-n", type=int, default=1, help="Self-consistency N for free-form")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Max generation tokens")
    parser.add_argument("--max-model-len", type=int, default=32768, help="vLLM max model length")
    parser.add_argument("--gpu-mem", type=float, default=0.90, help="GPU memory utilization")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (temp=0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Run evaluation pipeline."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Load data
    data_path = PROJECT_ROOT / args.data
    data = [json.loads(line) for line in open(data_path)]
    if args.subset:
        data = data[:args.subset]

    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_mem,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 65536,
    }
    if args.adapter:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)
    print("Model loaded.")

    # Build prompts
    def prompt_for_item(item: dict[str, Any]) -> str:
        system, user = build_prompt(item["question"], item.get("options"))
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    mcq_idx = [i for i, item in enumerate(data) if item.get("options")]
    free_idx = [i for i, item in enumerate(data) if not item.get("options")]

    def build_sampling_params(n: int) -> SamplingParams:
        if args.greedy and n == 1:
            temp, top_p, top_k = 0.0, 1.0, -1
        else:
            temp, top_p, top_k = args.temperature, 0.95, 40
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            n=n,
            seed=args.seed,
        )

    responses: list[str] = [""] * len(data)

    # MCQ pass
    if mcq_idx:
        mcq_prompts = [prompt_for_item(data[i]) for i in mcq_idx]
        sp = build_sampling_params(args.mcq_n)
        print(f"Generating {len(mcq_prompts)} MCQ (n={args.mcq_n}) ...")
        outs = llm.generate(mcq_prompts, sampling_params=sp)
        for local_i, global_i in enumerate(mcq_idx):
            if args.mcq_n > 1:
                responses[global_i] = pick_mcq_majority(outs[local_i].outputs)
            else:
                responses[global_i] = outs[local_i].outputs[0].text.strip()

    # Free-form pass
    if free_idx:
        free_prompts = [prompt_for_item(data[i]) for i in free_idx]
        sp = build_sampling_params(args.free_n)
        print(f"Generating {len(free_prompts)} free-form (n={args.free_n}) ...")
        outs = llm.generate(free_prompts, sampling_params=sp)
        for local_i, global_i in enumerate(free_idx):
            if args.free_n > 1:
                responses[global_i] = pick_freeform_majority(outs[local_i].outputs)
            else:
                responses[global_i] = outs[local_i].outputs[0].text.strip()

    # Score
    from judger import Judger
    judger = Judger(strict_extract=False)

    results: list[dict[str, Any]] = []
    correct_mcq = correct_free = total_mcq = total_free = 0

    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold = item["answer"]

        if is_mcq:
            correct = score_mcq(response, str(gold))
            total_mcq += 1
            correct_mcq += int(correct)
        else:
            correct = score_freeform(judger, response, gold)
            total_free += 1
            correct_free += int(correct)

        results.append({
            "id": item.get("id"),
            "is_mcq": is_mcq,
            "gold": gold,
            "response": response,
            "correct": correct,
        })

    # Print summary
    total_correct = correct_mcq + correct_free
    total = len(results)
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if total_mcq:
        print(f"  MCQ        : {correct_mcq:4d} / {total_mcq:4d}  ({correct_mcq / total_mcq * 100:.2f}%)")
    if total_free:
        print(f"  Free-form  : {correct_free:4d} / {total_free:4d}  ({correct_free / total_free * 100:.2f}%)")
    print(f"  Overall    : {total_correct:4d} / {total:4d}  ({total_correct / total * 100:.2f}%)")
    print("=" * 60)

    # Save results
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
