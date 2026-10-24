import time

from semantic_cache import SemanticCache


def fake_llm(prompt: str) -> str:
    time.sleep(0.5)  # pretend this is a slow, costly API call
    return f"[LLM answer to: {prompt}]"


cache = SemanticCache(threshold=0.75, ttl=3600)
ask = cache.cached(namespace="gpt-demo:v1")(fake_llm)

queries = [
    "What is the capital of France?",
    "what's the capital of france",      # near-duplicate -> should HIT
    "Capital city of France?",           # paraphrase -> may hit with a real embedder
    "Who wrote Hamlet?",                  # different topic -> MISS
    "Who wrote the play Hamlet?",         # should HIT
]
for q in queries:
    t = time.perf_counter()
    r = cache.get(q, "gpt-demo:v1")
    if r:
        print(f"HIT  ({r.score:.2f}) {time.perf_counter()-t:.3f}s  {q!r} ~ {r.matched_query!r}")
    else:
        ask(q)
        print(f"MISS        {time.perf_counter()-t:.3f}s  {q!r}")

# Namespaces keep answers separate (e.g. different model or system prompt)
print(
    "other namespace hit?",
    cache.get("What is the capital of France?", "another-model") is not None,
)
print(f"hit rate: {cache.stats.hit_rate:.0%}, entries: {cache.store.size()}")
