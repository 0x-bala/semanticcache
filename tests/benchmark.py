"""Benchmark script for semantic-cache.

Usage:
    cd "c:/Users/bmani/Downloads/Semantic cache"
    python tests/benchmark.py

Results are printed to stdout in a structured table.
No network access required.
"""
from __future__ import annotations

import pathlib
import statistics
import sys
import time

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

def _p99_ms(times: list[float]) -> float:
    return statistics.quantiles(times, n=100)[-1] * 1000

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

print("\nDone.")
