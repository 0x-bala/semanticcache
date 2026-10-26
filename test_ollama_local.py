"""Quick smoke test: semantic cache with a local Ollama LLM.

No server needed — just run this script directly.

Setup first:
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
"""

import time

try:
    import ollama
except ImportError:
    raise ImportError(
        "The ollama package is required for this test. "
        "Install it with: pip install -e '.[ollama]'"
    )

from semantic_cache import OllamaEmbedder, SemanticCache

LLM_MODEL = "llama3.2:3b"
EMBEDDER_MODEL = "nomic-embed-text"


def local_llm(prompt: str) -> str:
    r = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return r["response"]


def main() -> None:
    print(f"Connecting to Ollama (model={LLM_MODEL}, embedder={EMBEDDER_MODEL})...")
    embedder = OllamaEmbedder(model_name=EMBEDDER_MODEL)
    cache = SemanticCache(
        embedder=embedder,
        threshold=0.85,
        ttl=3600,
    )
    ask = cache.cached(namespace="ollama-test")(local_llm)

    queries = [
        "What is the capital of France?",
        "What's the capital of France?",       # near-duplicate → should HIT
        "Who wrote the play Hamlet?",           # different topic → MISS
        "Who wrote Hamlet?",                     # should HIT
    ]

    for q in queries:
        t = time.perf_counter()
        r = cache.get(q, "ollama-test")
        if r:
            matched = r.matched_query or "?"
            elapsed = time.perf_counter() - t
            print(f"  HIT  ({r.score:.2f}) {elapsed:.2f}s  {q!r} ~ {matched!r}")
        else:
            t0 = time.perf_counter()
            answer = ask(q)
            print(f"  MISS {time.perf_counter()-t0:.2f}s (cached) {q!r} -> {answer[:60]!r}...")

    print(f"\nHit rate: {cache.stats.hit_rate:.0%}, entries: {cache.store.size()}")
    cache.close()


if __name__ == "__main__":
    main()