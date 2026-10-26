"""Negative-case audit: measure similarity scores for pairs that must NOT hit.

Uses HashingEmbedder (offline, deterministic) so results are reproducible.
Run: python audit_negative_cases.py
"""

from semantic_cache import HashingEmbedder, SemanticCache, SQLiteStore
from semantic_cache.guard import check_match


def similarity(a: str, b: str, embedder: HashingEmbedder) -> float:
    va = embedder.embed(a)
    vb = embedder.embed(b)
    return float(va @ vb)


def main() -> None:
    embedder = HashingEmbedder(dim=1024)

    print("=== Negative-case similarity scores (HashingEmbedder, 1024-dim) ===\n")

    pairs = [
        # 1. Different entity, same template
        ("What is the capital of Germany?", "What is the capital of France?"),
        # 2. Different entity, same template
        ("Who wrote Macbeth?", "Who wrote Hamlet?"),
        # 3. Negation flip
        ("Is Paris the capital of France?", "Is Paris not the capital of France?"),
        # 4. Negation flip (contraction)
        ("Is Paris the capital of France?", "Isn't Paris the capital of France?"),
        # 5. Number difference
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

    for i, (a, b) in enumerate(pairs, 1):
        score = similarity(a, b, embedder)
        verdict = check_match(a, b).verdict.value
        flag = "MUST NOT HIT" if i <= 8 else "SHOULD HIT"
        print(f"{i:2d}. [{flag}] score={score:.4f}  guard={verdict}")
        print(f"     A: {a!r}")
        print(f"     B: {b!r}")

    # Now test through the actual cache
    print("\n=== Cache-level behaviour (threshold=0.85, guard on) ===\n")
    cache = SemanticCache(
        embedder=embedder,
        store=SQLiteStore(":memory:"),
        threshold=0.85,
    )

    # Store the "correct" answer
    cache.set("What is the capital of France?", "The capital of France is Paris.")
    cache.set("Who wrote Hamlet?", "The play Hamlet was written by William Shakespeare.")

    negative_queries = [
        "What is the capital of Germany?",
        "Who wrote Macbeth?",
        "Is Paris the capital of France?",
        "Is Paris not the capital of France?",
        "Convert 10 km to miles",
        "Who is the CEO of Tesla?",
        "Is x > 5?",
        "Enable the setting",
    ]

    for q in negative_queries:
        r = cache.get(q)
        status = "HIT (BUG)" if r else "MISS (correct)"
        score = r.score if r else 0.0
        print(f"  {status:20s} score={score:.4f}  {q!r}")

    # Empty / whitespace input
    print("\n=== Empty / whitespace input ===\n")
    for bad in ["", "   ", "\t\n"]:
        try:
            r = cache.get(bad)
            print(f"  get({bad!r}) -> {r}")
        except Exception as exc:
            print(f"  get({bad!r}) -> raised {type(exc).__name__}: {exc}")

    cache.close()


if __name__ == "__main__":
    main()