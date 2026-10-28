"""Tests for audit fixes: is_cacheable, whitespace validation, OllamaEmbedder init."""

from __future__ import annotations

import importlib.util

import pytest

from semantic_cache import HashingEmbedder, SemanticCache, SQLiteStore
from semantic_cache.cache import _default_is_cacheable

_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None

# ---------------------------------------------------------------------------
# is_cacheable
# ---------------------------------------------------------------------------


class TestIsCacheable:
    def test_default_rejects_empty(self) -> None:
        assert _default_is_cacheable("") is False

    def test_default_rejects_whitespace(self) -> None:
        assert _default_is_cacheable("   ") is False
        assert _default_is_cacheable("\t\n") is False

    def test_default_rejects_non_string(self) -> None:
        assert _default_is_cacheable(None) is False  # type: ignore[arg-type]
        assert _default_is_cacheable(42) is False  # type: ignore[arg-type]
        assert _default_is_cacheable([]) is False  # type: ignore[arg-type]

    def test_default_accepts_short_legitimate(self) -> None:
        assert _default_is_cacheable("42") is True
        assert _default_is_cacheable("Paris") is True

    def test_default_accepts_longer(self) -> None:
        assert _default_is_cacheable("hello world") is True

    def test_custom_predicate(self) -> None:
        def reject_errors(r: str) -> bool:
            return not r.startswith("ERROR:")

        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
            is_cacheable=reject_errors,
        )
        assert cache.is_cacheable("ok") is True
        assert cache.is_cacheable("ERROR: bad input") is False


# ---------------------------------------------------------------------------
# get_or_compute respects is_cacheable
# ---------------------------------------------------------------------------


class TestGetOrComputeIsCacheable:
    def test_bad_response_not_cached(self) -> None:
        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
            threshold=0.85,
        )

        def compute(_q: str) -> str:
            return "I don't know"  # blocked by default is_cacheable

        result = cache.get_or_compute("hello world", compute)
        assert result.hit is False
        assert result.response == "I don't know"
        assert cache.store.size() == 0  # nothing stored

    def test_error_string_not_cached(self) -> None:
        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
            threshold=0.85,
            is_cacheable=lambda r: not r.startswith("No question"),
        )

        def compute(_q: str) -> str:
            return "No question was provided."

        result = cache.get_or_compute("hello world", compute)
        assert result.hit is False
        assert cache.store.size() == 0

    def test_good_response_cached(self) -> None:
        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
            threshold=0.85,
        )

        def compute(_q: str) -> str:
            return "This is a good answer."

        r1 = cache.get_or_compute("hello world", compute)
        assert r1.hit is False
        r2 = cache.get_or_compute("hello world", compute)
        assert r2.hit is True  # second call should hit


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------


class TestQueryValidation:
    def test_set_rejects_empty_query(self) -> None:
        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
        )
        with pytest.raises(ValueError, match="empty or whitespace"):
            cache.set("", "answer")

    def test_set_rejects_whitespace_query(self) -> None:
        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
        )
        with pytest.raises(ValueError, match="empty or whitespace"):
            cache.set("   ", "answer")

    def test_set_rejects_tab_query(self) -> None:
        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
        )
        with pytest.raises(ValueError, match="empty or whitespace"):
            cache.set("\t\n", "answer")

    def test_get_returns_miss_for_empty_query(self) -> None:
        """get() must handle empty/whitespace gracefully (return None, not raise)."""
        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
        )
        assert cache.get("") is None

    def test_get_returns_miss_for_whitespace_query(self) -> None:
        cache = SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
        )
        assert cache.get("   ") is None
        assert cache.get("\t\n") is None


# ---------------------------------------------------------------------------
# OllamaEmbedder dimensions probe
# ---------------------------------------------------------------------------


class TestOllamaEmbedderDimensions:
    def test_probe_raises_when_ollama_unreachable(self) -> None:
        """OllamaEmbedder must fail closed: an unreachable Ollama raises at
        construction rather than silently guessing a dimension."""
        from semantic_cache.embedders import OllamaEmbedder

        with pytest.raises((ConnectionError, TimeoutError)):
            OllamaEmbedder(
                model_name="nomic-embed-text",
                base_url="http://127.0.0.1:1",  # closed port -> connection refused
                timeout=0.5,
            )

    def test_explicit_dimensions_skips_probe(self) -> None:
        from semantic_cache.embedders import OllamaEmbedder

        embedder = OllamaEmbedder(
            model_name="test-model",
            base_url="http://127.0.0.1:1",
            dimensions=512,
            timeout=0.5,
        )
        assert embedder.dimensions == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# CrossEncoderVerifier default threshold (regression)
# ---------------------------------------------------------------------------


class TestCrossEncoderVerifierThreshold:
    """The default threshold was 4.25 on a 0..5 scale, but
    cross-encoder/stsb-roberta-base returns 0..1, so the default was
    unreachable and the verifier never accepted anything. These tests use a
    mocked model so an unreachable default can never go unnoticed again.

    The four tests that monkeypatch ``sentence_transformers.CrossEncoder``
    skip when sentence-transformers is not installed (base install); the
    validation test below still runs because it only exercises the import
    guard inside CrossEncoderVerifier.__init__.
    """

    pytestmark = pytest.mark.skipif(
        not _HAS_SENTENCE_TRANSFORMERS,
        reason="needs sentence-transformers (not in the base dev install)",
    )

    def _verifier_with_mocked_model(self, monkeypatch, score: float):
        from semantic_cache.cross_encoder import CrossEncoderVerifier

        class _FakeModel:
            def predict(self, pairs, **kwargs):
                import numpy as np

                return np.asarray([score] * len(pairs))

        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            lambda *a, **kw: _FakeModel(),
        )
        return CrossEncoderVerifier()

    def test_default_threshold_is_0_9(self, monkeypatch) -> None:
        verifier = self._verifier_with_mocked_model(monkeypatch, 0.5)
        assert verifier.threshold == 0.9

    def test_default_accepts_score_at_threshold(self, monkeypatch) -> None:
        verifier = self._verifier_with_mocked_model(monkeypatch, 0.9)
        assert verifier("a", "b") is True

    def test_default_rejects_below_threshold(self, monkeypatch) -> None:
        verifier = self._verifier_with_mocked_model(monkeypatch, 0.5)
        assert verifier("a", "b") is False

    def test_default_rejects_non_finite_score(self, monkeypatch) -> None:
        import numpy as np

        from semantic_cache.cross_encoder import CrossEncoderVerifier

        class _NaNModel:
            def predict(self, pairs, **kwargs):
                return np.asarray([float("nan")] * len(pairs))

        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            lambda *a, **kw: _NaNModel(),
        )
        verifier = CrossEncoderVerifier()
        assert verifier("a", "b") is False

    def test_threshold_must_be_in_0_1(self, monkeypatch) -> None:
        from semantic_cache.cross_encoder import CrossEncoderVerifier

        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder",
            lambda *a, **kw: None,
        )
        with pytest.raises(ValueError, match="0.0 and 1.0"):
            CrossEncoderVerifier(threshold=4.25)