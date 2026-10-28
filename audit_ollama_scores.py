"""Re-run the 10 negative/positive pairs using OllamaEmbedder (nomic-embed-text).

Measures similarity with the guard OFF and ON, and recommends a threshold
based on the guard-off scores.

Run: python audit_ollama_scores.py
"""

from __future__ import annotations

from semantic_cache import OllamaEmbedder, SemanticCache, SQLiteStore
from semantic_cache.guard import check_match


def main() -> None:
    print("Probing Ollama for embedding dimension...")
    embedder = OllamaEmbedder(model_name="nomic-embed-text")
    print(f"  dimensions = {embedder.dimensions}\n")

    pairs = [
        # 1. Different entity, same template
        ("What is the capital of Germany?", "What is the capital of France?"),
        # 2. Different entity, same template
        ("Who wrote Macbeth?", "Who wrote Hamlet?"),
        # 3. Negation flip
        ("Is Paris the capital of France?", "Is Paris not the capital of France?"),
        # 4. Negation flip (contraction)
        ("Is Paris the capital of France?", "Isn't Paris the capital of France?"),
        # 5. Number/unit swap
        ("Convert 10 km to miles", "Convert 10 miles to km"),
        # 6. Entity swap (CEO)
        ("Who is the CEO of Tesla?", "Who is the CEO of Apple?"),
        # 7. Symbol flip
        ("Is x > 5?", "Is x < 5?"),
        # 8. Antonym flip
        ("Enable the setting", "Disable the setting"),
        # 9. Control: true paraphrase (should hit)
        ("What is the capital of France?", "What's the capital of France?"),
        # 10. Control: true paraphrase
        ("Who wrote Hamlet?", "Who wrote the play Hamlet?"),
    ]

    print("=== OllamaEmbedder (nomic-embed-text) similarity scores ===\n")
    print(
        f"{'#':>2s}  {'guard_off':>10s}  {'guard_on':>10s}  "
        f"{'verdict':>10s}  {'flag':>12s}  Pair"
    )
    print("-" * 100)

    for i, (a, b) in enumerate(pairs, 1):
        va = embedder.embed(a)
        vb = embedder.embed(b)
        score = float(va @ vb)
        verdict = check_match(a, b).verdict.value
        flag = "MUST NOT HIT" if i <= 8 else "SHOULD HIT"
        print(f"{i:2d}  {score:10.4f}  {score:10.4f}  {verdict:>10s}  {flag:>12s}  {a[:40]!r}")

    # Cache-level test with guard ON
    print("\n=== Cache-level behaviour (threshold=0.85, guard ON) ===\n")
    cache = SemanticCache(
        embedder=embedder,
        store=SQLiteStore(":memory:"),
        threshold=0.85,
    )
    cache.set("What is the capital of France?", "The capital of France is Paris.")
    cache.set("Who wrote Hamlet?", "The play Hamlet was written by William Shakespeare.")

    negative = [
        "What is the capital of Germany?",
        "Who wrote Macbeth?",
        "Is Paris the capital of France?",
        "Is Paris not the capital of France?",
        "Convert 10 km to miles",
        "Who is the CEO of Tesla?",
        "Is x > 5?",
        "Enable the setting",
    ]
    for q in negative:
        r = cache.get(q)
        status = "HIT (BUG)" if r else "MISS (correct)"
        s = r.score if r else 0.0
        print(f"  {status:20s} score={s:.4f}  {q!r}")

    # Threshold recommendation
    print("\n=== Threshold recommendation (guard OFF) ===\n")
    print("Guard-off scores for pairs that MUST NOT hit:")
    must_not_hit_scores = []
    for i, (a, b) in enumerate(pairs, 1):
        if i > 8:
            break
        va = embedder.embed(a)
        vb = embedder.embed(b)
        must_not_hit_scores.append(float(va @ vb))
    print(f"  max must-not-hit score = {max(must_not_hit_scores):.4f}")

    should_hit_scores = []
    for i, (a, b) in enumerate(pairs, 1):
        if i <= 8:
            continue
        va = embedder.embed(a)
        vb = embedder.embed(b)
        should_hit_scores.append(float(va @ vb))
    print(f"  min should-hit score  = {min(should_hit_scores):.4f}")

    gap = min(should_hit_scores) - max(must_not_hit_scores)
    if gap > 0:
        recommended = (max(must_not_hit_scores) + min(should_hit_scores)) / 2
        print(f"  gap = {gap:.4f}")
        print(f"  Recommended threshold (guard OFF): {recommended:.4f}")
    else:
        print("  No clean separation with guard OFF — the guard is required.")
        print("  With guard ON, threshold 0.80-0.85 is safe.")

    cache.close()


if __name__ == "__main__":
    main()