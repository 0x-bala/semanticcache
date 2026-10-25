"""Part 4: Shadow mode on real prompts (GLUE QQP dataset).

Runs shadow mode on a stream of 600 real prompts using SemanticCache:
- Seeds cache with 10k real questions from QQP.
- Replays a 50/50 mix of 300 duplicate queries and 300 non-duplicate queries.
- Runs with shadow_rate=1.0 (auditing every hit asynchronously).
- Evaluates estimated false-hit rate with Wilson 95% confidence intervals and 3/n rule.
- Compares shadow mode estimate to ground-truth false-hit rate from labeled dataset.
- Saves all mismatch records to benchmarks/results_real.json under part4_shadow_mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import types
from importlib.machinery import ModuleSpec
from typing import Any

# Ensure sklearn Cython DLL block on Windows App Control does not impede transformers
if "sklearn" not in sys.modules:
    m = types.ModuleType("sklearn")
    m.__spec__ = ModuleSpec("sklearn", None)
    sys.modules["sklearn"] = m
    m_met = types.ModuleType("sklearn.metrics")
    m_met.__spec__ = ModuleSpec("sklearn.metrics", None)
    m_met.roc_curve = lambda *a, **kw: (None, None, None)
    sys.modules["sklearn.metrics"] = m_met

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from semantic_cache.cache import SemanticCache  # noqa: E402
from semantic_cache.guard import check_match  # noqa: E402
from semantic_cache.store import SQLiteStore  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_real.json")


class LocalMiniLMEmbedder:
    """Local offline embedder using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        self.dim = 384
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            att_mask = inputs["attention_mask"].unsqueeze(-1)
            mask = att_mask.expand(outputs.last_hidden_state.size()).float()
            sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_emb / sum_mask
            normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normed.cpu().numpy().astype(np.float32)


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p_hat = k / n
    denominator = 1 + z**2 / n
    center_adj = p_hat + z**2 / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    lower = max(0.0, (center_adj - spread) / denominator)
    upper = min(1.0, (center_adj + spread) / denominator)
    return round(lower, 4), round(upper, 4)


def run_shadow_benchmark(stream_size: int = 600, seed_corpus_size: int = 10000) -> dict[str, Any]:
    """Execute shadow mode run on a stream of real queries."""
    print("=======================================================")
    print("PART 4: SHADOW MODE ON REAL PROMPTS")
    print(f"Stream Size: {stream_size} prompts | Seed Corpus: {seed_corpus_size} entries")
    print("=======================================================")

    pairs_file = os.path.join(DATA_DIR, "pairs_2k.json")
    corpus_file = os.path.join(DATA_DIR, "corpus_questions.json")
    corpus_vecs_file = os.path.join(DATA_DIR, "corpus_384.npy")

    if not (
        os.path.exists(pairs_file)
        and os.path.exists(corpus_file)
        and os.path.exists(corpus_vecs_file)
    ):
        raise FileNotFoundError("Prerequisite data files missing in benchmarks/data/")

    with open(pairs_file, encoding="utf-8") as f:
        pairs = json.load(f)
    with open(corpus_file, encoding="utf-8") as f:
        corpus = json.load(f)

    corpus_vecs = np.load(corpus_vecs_file)

    embedder = LocalMiniLMEmbedder()

    # Track mismatches explicitly
    mismatches = []

    def shadow_comparator(fresh: str, cached: str) -> bool:
        """Ground truth semantic comparator between live response and cached response."""
        return fresh.strip() == cached.strip()

    store = SQLiteStore(path=":memory:", embedding_dim=384, ann_backend="faiss", ann_threshold=5000)

    cache = SemanticCache(
        store=store,
        embedder=embedder,
        threshold=0.85,
        guard=True,
        verify_always=False,
        guard_buffer=0.05,
        shadow_rate=1.0,  # Audit 100% of hits
        shadow_max_concurrency=4,
        shadow_comparator=shadow_comparator,
    )

    # 1. Initialize store and seed with 10k real questions
    print(
        f"Seeding cache with {seed_corpus_size} real QQP questions "
        f"(embedder={cache._embedder_name})..."
    )
    # Seed with the first 10,000 corpus entries (which include the cached_q for the pairs)
    for i in range(seed_corpus_size):
        q = corpus[i]
        store.add(
            "default",
            q,
            corpus_vecs[i],
            f"Canonical response to: {q}",
            ttl=None,
            embedder=cache._embedder_name,
        )
    store.flush()

    # 2. Select 300 duplicates and 300 non-duplicates for the query stream
    dup_pairs = [p for p in pairs if p["is_duplicate"] == 1][: stream_size // 2]
    non_dup_pairs = [p for p in pairs if p["is_duplicate"] == 0][: stream_size // 2]

    stream = []
    for p in dup_pairs:
        stream.append({
            "query": p["query"],
            "target_cached_q": p["cached_q"],
            "ground_truth": 1,
            "oracle_response": f"Canonical response to: {p['cached_q']}",
        })
    for p in non_dup_pairs:
        stream.append({
            "query": p["query"],
            "target_cached_q": p["cached_q"],
            "ground_truth": 0,
            "oracle_response": f"Canonical response to: {p['query']}",
        })

    # Shuffle stream reproducibly
    rng = random.Random(42)
    rng.shuffle(stream)

    print(
        f"Replaying stream of {len(stream)} queries through SemanticCache with shadow_rate=1.0..."
    )
    total_queries = len(stream)
    hits = 0
    misses = 0

    ground_truth_true_hits = 0
    ground_truth_false_hits = 0

    t0 = time.perf_counter()
    for item in stream:
        q = item["query"]
        oracle_resp = item["oracle_response"]

        def compute_fn(_query: str, resp=oracle_resp) -> str:
            # Oracle compute returning canonical response for this query
            return resp

        res = cache.get_or_compute(q, compute=compute_fn)
        if res.hit:
            hits += 1
            if item["ground_truth"] == 1:
                ground_truth_true_hits += 1
            else:
                ground_truth_false_hits += 1
                # Record detailed mismatch
                matched = res.matched_query
                g_verdict = check_match(q, matched).verdict.value if matched else "UNKNOWN"
                mismatches.append({
                    "query": q,
                    "matched_query": res.matched_query,
                    "cached_response": res.response,
                    "live_response": oracle_resp,
                    "similarity": round(float(res.score), 4),
                    "guard_verdict": g_verdict,
                    "ground_truth_label": item["ground_truth"],
                    "verdict": "FALSE_HIT",
                })
        else:
            misses += 1

    # Wait for all background shadow tasks to complete
    cache.flush_shadow(timeout=15.0)
    elapsed = time.perf_counter() - t0

    stats = cache.stats
    shadow_evals = stats.shadow_evaluations
    shadow_matches = stats.shadow_matches
    shadow_mismatches = stats.shadow_mismatches
    shadow_errors = stats.shadow_errors

    est_false_hit_rate = stats.estimated_false_hit_rate
    wilson_low, wilson_high = wilson_score_interval(
        shadow_mismatches, max(1, shadow_matches + shadow_mismatches)
    )
    rule_of_three = round(3.0 / max(1, shadow_evals), 4) if shadow_mismatches == 0 else None

    true_false_hit_rate = round(ground_truth_false_hits / max(1, hits), 4)

    print("\n--- SHADOW MODE RESULTS ---")
    qps = round(total_queries / elapsed, 1)
    print(f"Total Stream Queries: {total_queries} in {elapsed:.2f}s ({qps} qps)")
    print(f"Cache Hits: {hits} | Misses: {misses}")
    print(f"Shadow Evaluations Triggered: {shadow_evals}")
    print(f"Shadow Matches: {shadow_matches}")
    print(f"Shadow Mismatches: {shadow_mismatches}")
    print(f"Shadow Errors: {shadow_errors}")
    print(
        f"Estimated False-Hit Rate: {est_false_hit_rate:.4f} "
        f"(Wilson 95% CI: [{wilson_low}, {wilson_high}])"
    )
    print(
        f"True False-Hit Rate (from labels): {true_false_hit_rate:.4f} "
        f"({ground_truth_false_hits} / {hits})"
    )
    print(f"Mismatches Captured: {len(mismatches)}")

    part4_data = {
        "status": "COMPLETED",
        "stream_size": total_queries,
        "seed_corpus_size": seed_corpus_size,
        "shadow_rate": 1.0,
        "total_queries": total_queries,
        "cache_hits": hits,
        "cache_misses": misses,
        "shadow_evaluations": shadow_evals,
        "shadow_matches": shadow_matches,
        "shadow_mismatches": shadow_mismatches,
        "shadow_errors": shadow_errors,
        "shadow_error_rate": round(shadow_errors / max(1, shadow_evals), 4),
        "estimated_false_hit_rate": round(est_false_hit_rate, 4),
        "wilson_95_ci": [wilson_low, wilson_high],
        "rule_of_three_bound": rule_of_three,
        "true_false_hit_count": ground_truth_false_hits,
        "true_false_hit_rate": true_false_hit_rate,
        "delta_estimate_vs_true": round(est_false_hit_rate - true_false_hit_rate, 4),
        "total_mismatch_records": len(mismatches),
        "mismatches": mismatches,
        "note": (
            "Audited with 100% shadow sampling (shadow_rate=1.0). "
            "Evaluated on real QQP prompt stream using canonical label-oracle comparator. "
            "Live Gemini generative LLM calls exhausted on Google Free Tier daily quota (HTTP 429)."
        ),
    }

    # Append to results_real.json
    results = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    results["part4_shadow_mode"] = part4_data
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nPart 4 results saved to {RESULTS_PATH}")

    store.close()
    return part4_data


def main():
    parser = argparse.ArgumentParser(description="Run shadow mode benchmark on real prompts.")
    parser.add_argument(
        "--stream-size", type=int, default=600, help="Number of real prompts in stream"
    )
    args = parser.parse_args()

    run_shadow_benchmark(stream_size=args.stream_size)


if __name__ == "__main__":
    main()
