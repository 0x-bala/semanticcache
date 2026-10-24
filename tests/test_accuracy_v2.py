"""Regression tests for the accuracy v2 patch; no network/model download required."""
from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from semantic_cache.cache import SemanticCache
from semantic_cache.cross_encoder import CrossEncoderVerifier
from semantic_cache.embedders import HashingEmbedder, SentenceTransformerEmbedder
from semantic_cache.gemini import GeminiEmbedder
from semantic_cache.guard import GuardVerdict, check_match, is_time_sensitive_query
from semantic_cache.store import SQLiteStore


class _MappedEmbedder:
    def __init__(self, vectors: dict[str, np.ndarray], dim: int = 4):
        self.vectors = vectors
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        return self.vectors[text].astype(np.float32, copy=True)


def _ranked_vec(score: float, axis: int, dim: int = 4) -> np.ndarray:
    result = np.zeros(dim, dtype=np.float32)
    result[0] = score
    result[axis] = math.sqrt(max(0.0, 1.0 - score * score))
    return result


def test_top_k_tries_second_candidate_after_top_one_is_rejected():
    query = "transfer 10 dollars from account 1 to account 2"
    wrong = "transfer 10 dollars from account 2 to account 1"
    equivalent = "please transfer 10 dollars from account 1 to account 2"
    vectors = {
        query: np.array([1, 0, 0, 0], dtype=np.float32),
        wrong: _ranked_vec(0.99, 1),
        equivalent: _ranked_vec(0.95, 2),
    }
    store = SQLiteStore(path=":memory:", embedding_dim=4)
    cache = SemanticCache(
        embedder=_MappedEmbedder(vectors),
        store=store,
        threshold=0.85,
        top_k=5,
    )
    cache.set(wrong, "unsafe answer")
    cache.set(equivalent, "correct reusable answer")

    result = cache.get(query)

    assert result is not None
    assert result.response == "correct reusable answer"
    assert result.matched_query == equivalent
    assert cache.stats.guard_rejects >= 1
    cache.close()


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("enable notifications", "disable notifications"),
        ("import the data", "export the data"),
        ("encrypt the backup", "decrypt the backup"),
        ("start the service", "stop the service"),
        ("allow access", "block access"),
    ],
)
def test_lexical_antonyms_are_rejected(left: str, right: str):
    assert check_match(left, right).verdict == GuardVerdict.REJECT


def test_swapped_number_order_is_rejected():
    result = check_match("convert 10 to 20", "convert 20 to 10")
    assert result.verdict == GuardVerdict.REJECT


def test_directional_preposition_order_is_rejected():
    result = check_match(
        "transfer funds from account 1 to account 2",
        "transfer funds to account 2 from account 1",
    )
    assert result.verdict == GuardVerdict.REJECT


def test_equivalent_linear_time_units_are_normalized():
    result = check_match("how long is 1 hour", "how long is 60 minutes")
    assert result.verdict == GuardVerdict.ACCEPT


def test_conversion_target_units_remain_distinct():
    result = check_match("convert 1 hour to minutes", "convert 1 hour to seconds")
    assert result.verdict == GuardVerdict.REJECT


def test_custom_synonym_groups_can_be_enabled_explicitly():
    a = "cancel subscription"
    b = "terminate membership"
    assert check_match(a, b).verdict != GuardVerdict.ACCEPT
    result = check_match(
        a,
        b,
        synonym_groups=[{"cancel", "terminate"}, {"subscription", "membership"}],
    )
    assert result.verdict == GuardVerdict.ACCEPT


def test_relative_time_queries_are_not_cached_or_reused():
    query = "What is the current price?"
    assert is_time_sensitive_query(query)
    assert check_match(query, query).verdict == GuardVerdict.REJECT

    store = SQLiteStore(path=":memory:", embedding_dim=1024)
    cache = SemanticCache(embedder=HashingEmbedder(), store=store)
    first = cache.get_or_compute(query, lambda _: "price A")
    second = cache.get_or_compute(query, lambda _: "price B")
    assert first.response == "price A"
    assert second.response == "price B"
    assert store.size() == 0
    cache.close()


def test_default_thresholds_are_embedder_specific_without_model_download():
    assert HashingEmbedder.default_threshold == 0.90
    assert SentenceTransformerEmbedder.default_threshold == 0.82
    assert GeminiEmbedder.default_threshold == 0.88

    cache = SemanticCache(embedder=HashingEmbedder(), store=SQLiteStore(path=":memory:"))
    assert cache.threshold == 0.90
    cache.close()


def test_cross_encoder_verifier_is_optional_and_uses_model_score(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

        def predict(self, pairs):
            return np.array([4.7 if pairs[0][0] == "same" else 2.0], dtype=np.float32)

    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    verifier = CrossEncoderVerifier(threshold=4.25)
    assert verifier("same", "pair") is True
    assert verifier("different", "pair") is False


def test_store_top_k_respects_similarity_floor_and_sort_order():
    embedder = _MappedEmbedder({
        "q": np.array([1, 0, 0, 0], dtype=np.float32),
        "candidate-1": _ranked_vec(0.91, 1),
        "candidate-2": _ranked_vec(0.97, 2),
    })
    store = SQLiteStore(path=":memory:", embedding_dim=4)
    for query in ("candidate-1", "candidate-2"):
        store.add(
            "default", query, embedder.embed(query), query, None,
            embedder="_MappedEmbedder:4",
        )
    candidates = store.search_candidates(
        "default",
        embedder.embed("q"),
        embedder="_MappedEmbedder:4",
        top_k=5,
        min_similarity=0.90,
    )
    assert [candidate.query for candidate in candidates] == ["candidate-2", "candidate-1"]
    assert store.search_candidates(
        "default",
        embedder.embed("q"),
        embedder="_MappedEmbedder:4",
        top_k=5,
        min_similarity=0.99,
    ) == []
    store.close()


def test_top_k_validation_rejects_unbounded_values():
    with pytest.raises(ValueError, match="must not exceed 500"):
        SemanticCache(top_k=501)

    store = SQLiteStore(path=":memory:", embedding_dim=4)
    with pytest.raises(ValueError, match="must not exceed 500"):
        store.search_candidates(
            "default", np.array([1, 0, 0, 0], dtype=np.float32), top_k=501
        )
    store.close()


def test_one_sided_directional_marker_difference_remains_verify():
    queries = (
        ("Summarize this article", "Summarize this article and translate it to French"),
        ("Summarize this article and translate it to French", "Summarize this article"),
    )
    for new_query, cached_query in queries:
        result = check_match(new_query, cached_query)
        assert result.verdict == GuardVerdict.VERIFY
