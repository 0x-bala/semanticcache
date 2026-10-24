"""Regression tests for safer semantic matching and paraphrase recall."""

from semantic_cache.cache import SemanticCache
from semantic_cache.embedders import HashingEmbedder
from semantic_cache.guard import GuardVerdict, check_match
from semantic_cache.store import SQLiteStore


def test_hashing_embedder_recovers_safe_paraphrase_at_production_threshold():
    embedder = HashingEmbedder(dim=1024)
    stored = embedder.embed("What is the capital of France?")
    paraphrase = embedder.embed("Capital city of France?")

    score = float(stored @ paraphrase)
    assert score >= 0.85, f"Expected content-focused similarity >= 0.85, got {score:.4f}"


def test_cache_hits_safe_paraphrase_without_lowering_threshold():
    store = SQLiteStore(path=":memory:", embedding_dim=1024)
    cache = SemanticCache(
        embedder=HashingEmbedder(dim=1024),
        store=store,
        threshold=0.85,
        guard=True,
    )
    cache.set("What is the capital of France?", "Paris")

    result = cache.get("Capital city of France?")

    assert result is not None
    assert result.hit is True
    assert result.response == "Paris"
    assert result.score >= 0.85
    cache.close()


def test_question_type_change_is_rejected():
    result = check_match(
        "Who is the capital of France?",
        "What is the capital of France?",
    )

    assert result.verdict == GuardVerdict.REJECT


def test_entity_substitution_is_not_accepted_as_safe_paraphrase():
    result = check_match(
        "What is the capital of Spain?",
        "What is the capital of France?",
    )

    assert result.verdict != GuardVerdict.ACCEPT


def test_contractions_use_consistent_embedding_normalization():
    embedder = HashingEmbedder(dim=1024)
    expanded = embedder.embed("What is the capital of France?")
    contracted = embedder.embed("What's the capital of France?")

    assert float(expanded @ contracted) > 0.999
