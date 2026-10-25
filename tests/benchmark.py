"""Benchmark script for semantic-cache.

Usage:
    cd "c:/Users/bmani/Downloads/Semantic cache"
    python tests/benchmark.py

Results are printed to stdout in a structured table.
No network access required.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np

from semantic_cache.cache import SemanticCache
from semantic_cache.embedders import HashingEmbedder
from semantic_cache.store import SQLiteStore

REPS = 500
DIM = 1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_ms(times: list[float]) -> float:
    return statistics.mean(times) * 1000

def _p50_ms(times: list[float]) -> float:
    return float(np.percentile(times, 50)) * 1000

def _p99_ms(times: list[float]) -> float:
    return float(np.percentile(times, 99)) * 1000

def bench(label: str, fn, n: int = REPS) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    print(f"  {label:<42}  mean={_mean_ms(times):7.3f}ms  p99={_p99_ms(times):7.3f}ms")
    return times

# ---------------------------------------------------------------------------
# Embedding benchmarks
# ---------------------------------------------------------------------------

print("\n=== Embedding latency ===")
emb = HashingEmbedder(dim=DIM)
short_text = "What is the capital of France?"
long_text = " ".join(["What is the capital of France?"] * 20)

bench("HashingEmbedder.embed() short text", lambda: emb.embed(short_text))
bench("HashingEmbedder.embed() long text",  lambda: emb.embed(long_text))

# ---------------------------------------------------------------------------
# Store benchmarks
# ---------------------------------------------------------------------------

print("\n=== SQLiteStore (in-memory, 100 entries) ===")

store_small = SQLiteStore(path=":memory:", embedding_dim=DIM)
queries, vecs = [], []
for i in range(100):
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    store_small.add("ns", f"query-{i}", v, f"response-{i}", ttl=None)
    queries.append(f"query-{i}")
    vecs.append(v)

probe = vecs[0]
bench("SQLiteStore.search() (100 entries)", lambda: store_small.search("ns", probe))
bench(
    "SQLiteStore.add()    (100 entries)",
    lambda: store_small.add("ns", "new-q", probe, "r", ttl=None),
)

print("\n=== SQLiteStore (in-memory, 1000 entries) ===")

store_large = SQLiteStore(path=":memory:", embedding_dim=DIM)
for i in range(1000):
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    store_large.add("ns", f"query-{i}", v, f"response-{i}", ttl=None)

bench("SQLiteStore.search() (1000 entries)", lambda: store_large.search("ns", probe))

# ---------------------------------------------------------------------------
# Full end-to-end cache benchmarks
# ---------------------------------------------------------------------------

print("\n=== SemanticCache end-to-end ===")

cache = SemanticCache(
    embedder=emb,
    store=SQLiteStore(path=":memory:", embedding_dim=DIM),
    threshold=0.85,
)
cache.set(short_text, "Paris")

bench("cache.get()  (HIT, warm index)",  lambda: cache.get(short_text))
bench("cache.get()  (MISS, warm index)", lambda: cache.get("Who wrote Hamlet?"))

fake_fn_calls = [0]
def fake_llm(p: str) -> str:
    fake_fn_calls[0] += 1
    return "answer"

bench(
    "cache.get_or_compute() (HIT, warm)",
    lambda: cache.get_or_compute(short_text, fake_llm),
)

print(f"\nfake_llm called {fake_fn_calls[0]} times during HIT benchmark (should be 0 or very few)")

print("\n=== Uncached vs cached comparison ===")
slow_fn_time_ms = 50  # simulated 50ms LLM call

def fake_slow_llm(p: str) -> str:
    time.sleep(slow_fn_time_ms / 1000)
    return "answer"

# Warm cache
cache2 = SemanticCache(
    embedder=emb,
    store=SQLiteStore(path=":memory:", embedding_dim=DIM),
    threshold=0.85,
)
cache2.set(short_text, "Paris")

n = 20
uncached_times = []
for _ in range(n):
    t0 = time.perf_counter()
    fake_slow_llm(short_text)
    uncached_times.append(time.perf_counter() - t0)

cached_times = []
for _ in range(n):
    t0 = time.perf_counter()
    cache2.get(short_text)
    cached_times.append(time.perf_counter() - t0)

speedup = _mean_ms(uncached_times) / _mean_ms(cached_times)
print(f"  Uncached mean: {_mean_ms(uncached_times):.1f}ms")
print(f"  Cached   mean: {_mean_ms(cached_times):.3f}ms")
print(f"  Speedup:       {speedup:.0f}x")

# ---------------------------------------------------------------------------
# Scale benchmarks (1k, 10k, 100k)
# ---------------------------------------------------------------------------

def run_scale_benchmarks(reps: int = 100) -> dict[str, Any]:
    print("\n=== Scale Lookup Latency Benchmark (1k, 10k, 100k) ===")
    scales = [1_000, 10_000, 100_000]
    results: dict[str, Any] = {}

    emb = HashingEmbedder(dim=DIM)
    store = SQLiteStore(path=":memory:", embedding_dim=DIM)
    cache = SemanticCache(
        embedder=emb,
        store=store,
        threshold=0.85,
    )

    exact_query = "What is the capital of France?"
    target_answer = "Paris"
    cache.set(exact_query, target_answer)

    semantic_query = "What is the capital of France please?"
    miss_query = "Quantum gravity superstrings in eleventh dimension"

    current_count = 1
    for scale in scales:
        scale_label = f"{scale // 1000}k"
        print(f"\nPopulating cache up to {scale:,} entries...")
        to_add = scale - current_count
        if to_add > 0:
            for i in range(current_count, scale):
                cache.set(f"Question about concept {i} in domain {i}", f"Answer {i}")
            current_count = scale

        # Warm-up / verify correctness
        res_exact = cache.get(exact_query)
        assert res_exact is not None and res_exact.hit, "Exact hit failed validation"
        res_semantic = cache.get(semantic_query)
        assert res_semantic is not None and res_semantic.hit, "Semantic hit failed validation"
        res_miss = cache.get(miss_query)
        assert res_miss is None, "Miss query unexpectedly matched"

        scale_results: dict[str, Any] = {}
        for lookup_type, q in [
            ("exact_hit", exact_query),
            ("semantic_hit", semantic_query),
            ("miss", miss_query),
        ]:
            times: list[float] = []
            for _ in range(reps):
                t0 = time.perf_counter()
                cache.get(q)
                times.append(time.perf_counter() - t0)

            scale_results[lookup_type] = {
                "p50_ms": round(_p50_ms(times), 4),
                "p99_ms": round(_p99_ms(times), 4),
                "mean_ms": round(_mean_ms(times), 4),
            }

        results[scale_label] = scale_results

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Scale':<10} {'Lookup Type':<16} {'p50 (ms)':<12} {'p99 (ms)':<12} {'Mean (ms)':<12}")
    print("-" * 65)
    for scale_label, metrics in results.items():
        for lookup_type, data in metrics.items():
            print(
                f"{scale_label:<10} {lookup_type:<16} "
                f"{data['p50_ms']:<12.4f} {data['p99_ms']:<12.4f} {data['mean_ms']:<12.4f}"
            )
    print("=" * 65)

    bench_dir = pathlib.Path(__file__).parent.parent / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_file = bench_dir / "results.json"
    payload = {
        "timestamp": time.time(),
        "embedder": "HashingEmbedder",
        "dim": DIM,
        "results": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved scale benchmark results to {out_file}")
    return results


run_scale_benchmarks(reps=100)

print("\nDone.")
