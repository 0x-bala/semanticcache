"""Which guard rules block true duplicates, and which actually stop wrong answers?

For every labeled pair this records the guard verdict and rule, then prints, per rule,
how many true duplicates (label 1) and how many non-duplicates (label 0) it fires on.
The pair set is balanced, so a rule whose "dup share" is well above 50% blocks more
good paraphrases than bad matches and is a candidate to relax (REJECT -> VERIFY).
A rule with a low dup share is doing its job.

Usage:
    python benchmarks/guard_rules.py            # real QQP pairs (needs benchmarks/data)
No network or API key is needed.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from semantic_cache.guard import check_match  # noqa: E402


def rule_name(reason: str) -> str:
    return reason.split(":")[0].strip()


def rule_table(pairs: list[dict[str, Any]], synonym_groups=None) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], Counter] = {}
    for p in pairs:
        res = check_match(p["query"], p["cached_q"], synonym_groups=synonym_groups)
        key = (res.verdict.value, rule_name(res.reason))
        counts.setdefault(key, Counter())[int(p["is_duplicate"])] += 1
    rows = []
    for (verdict, rule), c in counts.items():
        dup, non = c[1], c[0]
        rows.append(
            {
                "verdict": verdict,
                "rule": rule,
                "on_duplicates": dup,
                "on_non_duplicates": non,
                "dup_share": round(dup / (dup + non), 3),
            }
        )
    rows.sort(key=lambda r: (r["verdict"], -(r["on_duplicates"] + r["on_non_duplicates"])))
    return rows


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from real_embeddings import load_or_prep_dataset  # noqa: PLC0415

    pairs, _ = load_or_prep_dataset()
    rows = rule_table(pairs)
    print(f"{'verdict':8} {'rule':50} {'dup':>5} {'non-dup':>8} {'dup share':>10}")
    for r in rows:
        print(
            f"{r['verdict']:8} {r['rule'][:50]:50} {r['on_duplicates']:>5} "
            f"{r['on_non_duplicates']:>8} {r['dup_share']:>10.2f}"
        )
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard_rules.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
