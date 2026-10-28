"""Latency probe against a running Ollama.

Measures warm single-embed latency and CrossEncoderVerifier scoring latency.
Does not change any library behaviour. Run:

    python benchmarks/latency_probe.py
"""

from __future__ import annotations

import json
import statistics
import time

from semantic_cache.cross_encoder import CrossEncoderVerifier
from semantic_cache.embedders import OllamaEmbedder

PAIRS = [
    ("Who wrote Hamlet?", "Who is the author of Hamlet?"),
    ("What is the capital of France?", "Capital city of France?"),
    ("How tall is Mount Everest?", "What is the height of Mount Everest?"),
    ("Who founded Apple?", "Which person started Apple?"),
    ("What is the boiling point of water in Celsius?",
     "At what temperature in degrees Celsius does water boil?"),
    ("Who wrote Hamlet?", "Can you tell me who wrote the play Hamlet?"),
    ("Who wrote Macbeth?", "Who wrote Hamlet?"),
    ("What is the capital of Germany?", "What is the capital of France?"),
    ("How tall is K2?", "How tall is Mount Everest?"),
    ("Who founded Microsoft?", "Who founded Apple?"),
    ("What is the boiling point of water in Fahrenheit?",
     "What is the boiling point of water in Celsius?"),
    ("Who founded Apple in 2020?", "Who founded Apple?"),
    ("Where is the Eiffel Tower?", "In which city is the Eiffel Tower located?"),
    ("What is the capital of Japan?", "Which city is the capital of Japan?"),
    ("How many days are in a leap year?", "What is the number of days in a leap year?"),
    ("Enable the setting", "Disable the setting"),
    ("Is x > 5?", "Is x < 5?"),
    ("Convert 10 km to miles", "Convert 10 miles to km"),
    ("What is C++?", "What is C#?"),
    ("Who is the CEO of Tesla?", "Who is the CEO of Apple?"),
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def main() -> None:
    print("Embedding warm-up (ignored in timing)...")
    embedder = OllamaEmbedder(model_name="nomic-embed-text")
    print(f"  dimensions={embedder.dimensions} keep_alive={embedder.keep_alive}")

    # Warm the connection and cache the model in Ollama's process.
    embedder.embed("warmup")
    time.sleep(0.5)

    print("\nMeasuring 50 warm single embed calls...")
    latencies_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        embedder.embed("The quick brown fox jumps over the lazy dog")
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    latencies_ms.sort()
    print(f"  p50 = {percentile(latencies_ms, 50):.2f} ms")
    print(f"  p95 = {percentile(latencies_ms, 95):.2f} ms")
    print(f"  min = {latencies_ms[0]:.2f} ms  max = {latencies_ms[-1]:.2f} ms")

    print("\nCrossEncoderVerifier scoring 20 pairs...")
    verifier = CrossEncoderVerifier(threshold=0.9)
    scores: list[float] = []
    times_ms: list[float] = []
    for a, b in PAIRS:
        t0 = time.perf_counter()
        s = verifier.model.predict([[a, b]])[0]
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        scores.append(float(s))
    times_ms.sort()
    print(f"  per-pair p50 = {percentile(times_ms, 50):.2f} ms")
    print(f"  per-pair p95 = {percentile(times_ms, 95):.2f} ms")
    print(f"  score range = {min(scores):.4f} .. {max(scores):.4f}")
    print(f"  accepted at 0.9 = {sum(1 for s in scores if s >= 0.9)}/{len(scores)}")

    print("\nPer-pair scores:")
    for (a, b), s in zip(PAIRS, scores):
        print(f"  {s:6.4f}  {a[:45]!r:47s} vs {b[:45]!r}")

    print("\nDone.")


if __name__ == "__main__":
    main()