"""Evaluation harness for fine-tuned models.

Loads the base Qwen3-4B model with an optional LoRA adapter via vLLM and runs
inference + scoring on the public dataset. Reports accuracy by question type.

This is the main evaluation entry point for Phase 2 and Phase 3 checkpoints.

Usage:
    # Evaluate base model (no adapter)
    python scripts/evaluate.py

    # Evaluate SFT checkpoint
    python scripts/evaluate.py --adapter checkpoints/sft/final

    # Evaluate GRPO checkpoint
    python scripts/evaluate.py --adapter checkpoints/grpo/final

    # Quick smoke test
    python scripts/evaluate.py --subset 20 --adapter checkpoints/sft/final
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Prompt construction (mirrors notebook) ────────────────────────────────────

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


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    """Extract a single letter from \\boxed{} or fallback to last capital."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def extract_boxed_answer(text: str) -> str:
    """Extract raw content inside the last \\boxed{} after </think>."""
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


def pick_mcq_majority(outputs: list[Any]) -> str:
    """Majority vote on boxed letter."""
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
    """Majority vote on boxed answer string."""
    outs = list(outputs)
    answers = [extract_boxed_answer(o.text.strip()) for o in outs]
    valid = [(a, i) for i, a in enumerate(answers) if a]
    if not valid:
        return outs[0].text.strip()
    normalized = [re.sub(r"\s+", " ", a).strip().lower() for a, _ in valid]
    counter = Counter(normalized)
    winner_norm, _ = counter.most_common(1)[0]
    for (a, idx), norm in zip(valid, normalized):
        if norm == winner_norm:
            return outs[idx].text.strip()
    return outs[0].text.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned math model")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--output", default="results/eval_results.jsonl")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--mcq-n", type=int, default=1, help="MCQ self-consistency samples")
    parser.add_argument("--free-n", type=int, default=1, help="Free-form self-consistency samples")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the full evaluation pipeline."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Load data
    data_path = PROJECT_ROOT / args.data
    data = [json.loads(line) for line in open(data_path)]
    if args.subset:
        data = data[:args.subset]
    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build vLLM engine
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

    lora_request = None
    if args.adapter:
        adapter_path = str(PROJECT_ROOT / args.adapter)
        lora_request = LoRARequest("math_adapter", 1, adapter_path)
        print(f"Using LoRA adapter: {adapter_path}")

    # Build prompts
    def prompt_for_item(item: dict[str, Any]) -> str:
        system, user = build_prompt(item["question"], item.get("options"))
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    mcq_idx = [i for i, d in enumerate(data) if d.get("options")]
    free_idx = [i for i, d in enumerate(data) if not d.get("options")]

    def build_sp(n: int) -> SamplingParams:
        if args.greedy and n == 1:
            return SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, n=1, seed=args.seed)
        return SamplingParams(
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=0.95, top_k=40, n=n, seed=args.seed,
        )

    responses: list[str] = [""] * len(data)

    gen_kwargs: dict[str, Any] = {}
    if lora_request:
        gen_kwargs["lora_request"] = lora_request

    # MCQ pass
    if mcq_idx:
        prompts = [prompt_for_item(data[i]) for i in mcq_idx]
        sp = build_sp(args.mcq_n)
        print(f"Generating {len(prompts)} MCQ (n={args.mcq_n}) ...")
        outs = llm.generate(prompts, sampling_params=sp, **gen_kwargs)
        for li, gi in enumerate(mcq_idx):
            responses[gi] = pick_mcq_majority(outs[li].outputs) if args.mcq_n > 1 else outs[li].outputs[0].text.strip()

    # Free-form pass
    if free_idx:
        prompts = [prompt_for_item(data[i]) for i in free_idx]
        sp = build_sp(args.free_n)
        print(f"Generating {len(prompts)} free-form (n={args.free_n}) ...")
        outs = llm.generate(prompts, sampling_params=sp, **gen_kwargs)
        for li, gi in enumerate(free_idx):
            responses[gi] = pick_freeform_majority(outs[li].outputs) if args.free_n > 1 else outs[li].outputs[0].text.strip()

    # Score
    from judger import Judger
    judger = Judger(strict_extract=False)

    results: list[dict[str, Any]] = []
    correct_mcq = correct_free = total_mcq = total_free = 0

    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold = item["answer"]
        if is_mcq:
            correct = extract_letter(response) == str(gold).strip().upper()
            total_mcq += 1
            correct_mcq += int(correct)
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list))
            except Exception:
                correct = False
            total_free += 1
            correct_free += int(correct)

        results.append({
            "id": item.get("id"), "is_mcq": is_mcq, "gold": gold,
            "response": response, "correct": correct,
        })

    # Print summary
    total_correct = correct_mcq + correct_free
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if total_mcq:
        print(f"  MCQ        : {correct_mcq:4d} / {total_mcq:4d}  ({correct_mcq / total_mcq * 100:.2f}%)")
    if total_free:
        print(f"  Free-form  : {correct_free:4d} / {total_free:4d}  ({correct_free / total_free * 100:.2f}%)")
    print(f"  Overall    : {total_correct:4d} / {len(results):4d}  ({total_correct / len(results) * 100:.2f}%)")
    print("=" * 60)

    # Save
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
