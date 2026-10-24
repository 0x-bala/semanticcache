"""Real-data threshold tuning tool for SemanticCache.

Evaluates precision, recall, and F1 across candidate similarity thresholds
with the hit guard ON vs OFF, and determines the lowest threshold with zero
wrong hits (false positives).

Usage:
    python tools/tune_threshold.py --csv path/to/dataset.csv
    python tools/tune_threshold.py  # uses built-in benchmark dataset
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
from typing import NamedTuple

import numpy as np

# Add repository root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from semantic_cache.embedders import HashingEmbedder, validate_embedding
from semantic_cache.guard import GuardVerdict, check_match


class LabeledPair(NamedTuple):
    query_a: str
    query_b: str
    should_hit: bool


DEFAULT_DATASET = [
    # Legitimate paraphrases (should_hit = True)
    LabeledPair("what's the capital of france", "What is the capital of France?", True),
    LabeledPair("How can I reset my password?", "How do I reset my password?", True),
    LabeledPair("Who wrote the play Hamlet?", "Who wrote Hamlet?", True),
    LabeledPair("Capital city of France?", "What is the capital of France?", True),
    LabeledPair("official currency of Japan", "What is the official currency of Japan?", True),
    LabeledPair("Where is the Eiffel Tower?", "Where is the Eiffel Tower located?", True),
    # Entity swaps / slot substitutions (should_hit = False)
    LabeledPair("What is the capital of Spain?", "What is the capital of France?", False),
    LabeledPair("How do I reset my router?", "How do I reset my password?", False),
    LabeledPair("Who wrote Macbeth?", "Who wrote Hamlet?", False),
    LabeledPair("Who is the CEO of Apple?", "Who is the CEO of Tesla?", False),
    # Unit / order inversions (should_hit = False)
    LabeledPair("Convert 10 miles to km", "Convert 10 km to miles", False),
    LabeledPair("Convert 50 USD to EUR", "Convert 50 EUR to USD", False),
    # Negations & prefix flips (should_hit = False)
    LabeledPair("refundable ticket policy", "non-refundable ticket policy", False),
    LabeledPair("hotel allows pets", "hotel does not allow pets", False),
    LabeledPair("unlimited cloud storage", "limited cloud storage", False),
]


def load_dataset_from_csv(csv_path: str) -> list[LabeledPair]:
    """Load prompt pairs from a CSV file (query_a, query_b, should_hit)."""
    pairs: list[LabeledPair] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_a = row.get("query_a") or row.get("query1") or row.get("new_query") or ""
            q_b = row.get("query_b") or row.get("query2") or row.get("cached_query") or ""
            label_val = row.get("should_hit") or row.get("label") or row.get("match") or "false"
            should_hit = str(label_val).strip().lower() in ("true", "1", "yes", "y", "hit")
            if q_a and q_b:
                pairs.append(LabeledPair(q_a.strip(), q_b.strip(), should_hit))
    return pairs


class MetricRow(NamedTuple):
    threshold: float
    precision_no_guard: float
    recall_no_guard: float
    f1_no_guard: float
    wrong_hits_no_guard: int
    precision_guard: float
    recall_guard: float
    f1_guard: float
    wrong_hits_guard: int


def evaluate_dataset(
    pairs: list[LabeledPair],
    embedder=None,
    step: float = 0.02,
) -> tuple[list[MetricRow], float | None, float | None]:
    """Evaluate accuracy across thresholds [0.70 .. 0.98] with and without guard."""
    if embedder is None:
        embedder = HashingEmbedder(dim=1024)

    # Pre-embed and pre-compute similarity & guard verdict
    cached_evals = []
    for pair in pairs:
        v_a = validate_embedding(embedder.embed(pair.query_a))
        v_b = validate_embedding(embedder.embed(pair.query_b))
        sim = float(v_a @ v_b)
        guard_res = check_match(pair.query_a, pair.query_b)
        cached_evals.append((pair, sim, guard_res.verdict))

    thresholds = [round(t, 4) for t in np.arange(0.70, 0.985, step)]
    rows: list[MetricRow] = []

    lowest_zero_wrong_no_guard: float | None = None
    lowest_zero_wrong_guard: float | None = None

    for t in thresholds:
        tp_ng = fp_ng = fn_ng = 0
        tp_g = fp_g = fn_g = 0

        for pair, sim, verdict in cached_evals:
            # Without guard
            hit_ng = sim >= t
            if hit_ng:
                if pair.should_hit:
                    tp_ng += 1
                else:
                    fp_ng += 1
            else:
                if pair.should_hit:
                    fn_ng += 1

            # With guard
            hit_g = (sim >= t) and (verdict == GuardVerdict.ACCEPT)
            if hit_g:
                if pair.should_hit:
                    tp_g += 1
                else:
                    fp_g += 1
            else:
                if pair.should_hit:
                    fn_g += 1

        p_ng = tp_ng / (tp_ng + fp_ng) if (tp_ng + fp_ng) > 0 else 1.0
        r_ng = tp_ng / (tp_ng + fn_ng) if (tp_ng + fn_ng) > 0 else 0.0
        f1_ng = (2 * p_ng * r_ng) / (p_ng + r_ng) if (p_ng + r_ng) > 0 else 0.0

        p_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 1.0
        r_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
        f1_g = (2 * p_g * r_g) / (p_g + r_g) if (p_g + r_g) > 0 else 0.0

        if fp_ng == 0 and lowest_zero_wrong_no_guard is None:
            lowest_zero_wrong_no_guard = t
        if fp_g == 0 and lowest_zero_wrong_guard is None:
            lowest_zero_wrong_guard = t

        rows.append(
            MetricRow(
                threshold=t,
                precision_no_guard=p_ng,
                recall_no_guard=r_ng,
                f1_no_guard=f1_ng,
                wrong_hits_no_guard=fp_ng,
                precision_guard=p_g,
                recall_guard=r_g,
                f1_guard=f1_g,
                wrong_hits_guard=fp_g,
            )
        )

    return rows, lowest_zero_wrong_no_guard, lowest_zero_wrong_guard


def print_report(
    rows: list[MetricRow],
    lowest_ng: float | None,
    lowest_g: float | None,
    total_pairs: int,
) -> None:
    """Print ASCII comparison table and recommendation summary."""
    print("=" * 84)
    print(f" THRESHOLD TUNING REPORT ({total_pairs} prompt pairs)")
    print("=" * 84)
    header = (
        f"{'Thresh':<7} | "
        f"{'Prec (Off)':<10} {'Rec (Off)':<10} {'Wrong':<6} | "
        f"{'Prec (On)':<10} {'Rec (On)':<10} {'F1 (On)':<8} {'Wrong':<6}"
    )
    print(header)
    print("-" * 84)

    for r in rows:
        part1 = (
            f"{r.threshold:<7.2f} | "
            f"{r.precision_no_guard:<10.2%} {r.recall_no_guard:<10.2%} "
            f"{r.wrong_hits_no_guard:<6d} | "
        )
        part2 = (
            f"{r.precision_guard:<10.2%} {r.recall_guard:<10.2%} "
            f"{r.f1_guard:<8.2%} {r.wrong_hits_guard:<6d}"
        )
        print(part1 + part2)

    print("=" * 84)
    print(" SUMMARY & RECOMMENDATIONS")
    print("=" * 84)
    lowest_ng_str = f"{lowest_ng:.2f}" if lowest_ng is not None else "None found"
    lowest_g_str = f"{lowest_g:.2f}" if lowest_g is not None else "None found"
    print(f" Lowest threshold with 0 wrong hits WITHOUT guard : {lowest_ng_str}")
    print(f" Lowest threshold with 0 wrong hits WITH guard    : {lowest_g_str}")
    if lowest_g is not None:
        rec_low = max(0.88, lowest_g)
        rec_high = min(0.95, rec_low + 0.04)
        print(f" Recommended operational threshold range          : {rec_low:.2f} - {rec_high:.2f}")
    print("=" * 84)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune semantic cache threshold")
    parser.add_argument("--csv", type=str, default="", help="Path to CSV dataset")
    parser.add_argument("--step", type=float, default=0.02, help="Threshold step size")
    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        print(f"Loading dataset from: {args.csv}")
        dataset = load_dataset_from_csv(args.csv)
    else:
        print("Using built-in benchmark dataset (15 labeled test pairs)")
        dataset = DEFAULT_DATASET

    rows, lowest_ng, lowest_g = evaluate_dataset(dataset, step=args.step)
    print_report(rows, lowest_ng, lowest_g, len(dataset))


if __name__ == "__main__":
    main()
