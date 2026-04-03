"""Error analysis script for math reasoning evaluation results.

Reads evaluation results and the original dataset, then categorizes failures
by question type, answer format, and failure mode. Outputs a summary report
to help guide prompt/training iteration.

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --results results/eval_results.jsonl
    python scripts/error_analysis.py --results results/eval_results.jsonl --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_boxed_content(text: str) -> str:
    """Extract content inside the last \\boxed{} after </think>."""
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


def classify_failure(result: dict[str, Any]) -> str:
    """Classify a failure into a category based on the response content."""
    response = result.get("response", "")

    if not response or len(response.strip()) < 10:
        return "empty_or_very_short"

    boxed = extract_boxed_content(response)
    if not boxed:
        if "\\boxed" in response:
            return "malformed_boxed"
        return "no_boxed_answer"

    if result["is_mcq"]:
        letter_match = re.match(r"^[A-Za-z]$", boxed.strip())
        if not letter_match:
            return "mcq_non_letter_in_boxed"
        return "mcq_wrong_letter"

    return "wrong_answer"


def estimate_response_length(response: str) -> str:
    """Bin response length into categories."""
    think_end = response.rfind("</think>")
    reasoning = response[:think_end] if think_end >= 0 else response
    word_count = len(reasoning.split())
    if word_count < 50:
        return "very_short (<50 words)"
    if word_count < 200:
        return "short (50-200)"
    if word_count < 500:
        return "medium (200-500)"
    if word_count < 1000:
        return "long (500-1000)"
    return "very_long (>1000)"


def analyze_results(
    results: list[dict[str, Any]],
    data: list[dict[str, Any]],
    verbose: bool = False,
) -> None:
    """Print comprehensive error analysis report."""
    # Build lookup from data
    data_by_id = {d["id"]: d for d in data}

    total = len(results)
    correct = sum(r["correct"] for r in results)
    incorrect = total - correct

    mcq_results = [r for r in results if r["is_mcq"]]
    free_results = [r for r in results if not r["is_mcq"]]
    mcq_correct = sum(r["correct"] for r in mcq_results)
    free_correct = sum(r["correct"] for r in free_results)

    print("=" * 70)
    print("ERROR ANALYSIS REPORT")
    print("=" * 70)

    # Overall accuracy
    print(f"\n{'Overall':.<40} {correct:4d} / {total:4d} ({correct / total * 100:.2f}%)")
    if mcq_results:
        print(f"{'  MCQ':.<40} {mcq_correct:4d} / {len(mcq_results):4d} ({mcq_correct / len(mcq_results) * 100:.2f}%)")
    if free_results:
        print(f"{'  Free-form':.<40} {free_correct:4d} / {len(free_results):4d} ({free_correct / len(free_results) * 100:.2f}%)")

    # Failure mode breakdown
    failures = [r for r in results if not r["correct"]]
    if not failures:
        print("\nNo failures to analyze!")
        return

    print(f"\n{'─' * 70}")
    print("FAILURE MODE BREAKDOWN")
    print(f"{'─' * 70}")

    failure_modes = Counter(classify_failure(r) for r in failures)
    for mode, count in failure_modes.most_common():
        pct = count / incorrect * 100
        print(f"  {mode:.<45} {count:4d} ({pct:.1f}%)")

    # Response length distribution for failures vs. successes
    print(f"\n{'─' * 70}")
    print("RESPONSE LENGTH DISTRIBUTION")
    print(f"{'─' * 70}")

    correct_results = [r for r in results if r["correct"]]
    correct_lengths = Counter(estimate_response_length(r["response"]) for r in correct_results)
    failure_lengths = Counter(estimate_response_length(r["response"]) for r in failures)

    all_bins = sorted(set(list(correct_lengths.keys()) + list(failure_lengths.keys())))
    print(f"  {'Length bin':<30} {'Correct':>10} {'Incorrect':>10}")
    print(f"  {'─' * 50}")
    for bin_name in all_bins:
        c = correct_lengths.get(bin_name, 0)
        f = failure_lengths.get(bin_name, 0)
        print(f"  {bin_name:<30} {c:>10} {f:>10}")

    # MCQ answer distribution (for failures)
    if mcq_results:
        print(f"\n{'─' * 70}")
        print("MCQ ANALYSIS")
        print(f"{'─' * 70}")

        mcq_failures = [r for r in mcq_results if not r["correct"]]
        if mcq_failures:
            predicted = Counter()
            gold_dist = Counter()
            for r in mcq_failures:
                boxed = extract_boxed_content(r["response"])
                letter = boxed.strip().upper() if len(boxed.strip()) == 1 else "?"
                predicted[letter] += 1
                gold_letter = str(r["gold"]).strip().upper()
                gold_dist[gold_letter] += 1

            print("  Predicted letter distribution (failures):")
            for letter, count in predicted.most_common():
                print(f"    {letter}: {count}")

            print("  Gold letter distribution (failures):")
            for letter, count in gold_dist.most_common():
                print(f"    {letter}: {count}")

    # Free-form answer count analysis
    if free_results:
        print(f"\n{'─' * 70}")
        print("FREE-FORM ANALYSIS")
        print(f"{'─' * 70}")

        multi_answer = [r for r in free_results if isinstance(r["gold"], list) and len(r["gold"]) > 1]
        single_answer = [r for r in free_results if not isinstance(r["gold"], list) or len(r["gold"]) == 1]

        if multi_answer:
            ma_correct = sum(r["correct"] for r in multi_answer)
            print(f"  Multi-answer questions: {ma_correct} / {len(multi_answer)} ({ma_correct / len(multi_answer) * 100:.1f}%)")

        if single_answer:
            sa_correct = sum(r["correct"] for r in single_answer)
            print(f"  Single-answer questions: {sa_correct} / {len(single_answer)} ({sa_correct / len(single_answer) * 100:.1f}%)")

        # Count by number of sub-answers
        by_n_answers: dict[int, list[dict]] = defaultdict(list)
        for r in free_results:
            gold = r["gold"]
            n = len(gold) if isinstance(gold, list) else 1
            by_n_answers[n].append(r)

        print("\n  Accuracy by number of sub-answers:")
        for n in sorted(by_n_answers.keys()):
            subset = by_n_answers[n]
            c = sum(r["correct"] for r in subset)
            print(f"    {n} answer(s): {c:4d} / {len(subset):4d} ({c / len(subset) * 100:.1f}%)")

    # Verbose: show sample failures
    if verbose:
        print(f"\n{'─' * 70}")
        print("SAMPLE FAILURES (first 10)")
        print(f"{'─' * 70}")

        for r in failures[:10]:
            qid = r["id"]
            qtype = "MCQ" if r["is_mcq"] else "Free"
            mode = classify_failure(r)
            boxed = extract_boxed_content(r["response"])
            gold = r["gold"]

            print(f"\n  [ID {qid}] Type={qtype} Mode={mode}")
            print(f"    Gold:      {gold}")
            print(f"    Predicted: {boxed or '(none)'}")

            # Show question text if available
            orig = data_by_id.get(qid, {})
            question = orig.get("question", "")
            if question:
                print(f"    Question:  {question[:150]}{'...' if len(question) > 150 else ''}")

    # Actionable recommendations
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 70}")

    if failure_modes.get("no_boxed_answer", 0) > incorrect * 0.1:
        print("  [HIGH] Many responses lack \\boxed{} -- strengthen format instructions in prompt")

    if failure_modes.get("empty_or_very_short", 0) > incorrect * 0.05:
        print("  [HIGH] Some responses are empty/very short -- check max_tokens and model loading")

    if failure_modes.get("mcq_non_letter_in_boxed", 0) > 5:
        print("  [MED]  MCQ responses put non-letters in \\boxed{} -- reinforce MCQ prompt")

    if failure_modes.get("mcq_wrong_letter", 0) > len(mcq_results) * 0.3:
        print("  [MED]  High MCQ error rate -- try self-consistency voting (--mcq-n 8)")

    if failure_modes.get("wrong_answer", 0) > incorrect * 0.5:
        print("  [INFO] Most failures are wrong answers (reasoning errors) -- SFT/GRPO will help")

    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze math evaluation errors")
    parser.add_argument("--results", default="results/eval_results.jsonl", help="Results JSONL")
    parser.add_argument("--data", default="data/public.jsonl", help="Original dataset")
    parser.add_argument("--verbose", action="store_true", help="Show sample failures")
    return parser.parse_args()


def main() -> None:
    """Run error analysis."""
    args = parse_args()

    results_path = PROJECT_ROOT / args.results
    data_path = PROJECT_ROOT / args.data

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run evaluation first: python scripts/evaluate.py")
        sys.exit(1)

    results = [json.loads(line) for line in open(results_path)]
    data = [json.loads(line) for line in open(data_path)]

    analyze_results(results, data, verbose=args.verbose)


if __name__ == "__main__":
    main()
