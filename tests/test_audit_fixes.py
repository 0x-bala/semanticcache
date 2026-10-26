"""Tests for audit fixes: is_cacheable, whitespace validation, OllamaEmbedder init."""

from __future__ import annotations

import pytest

from semantic_cache import HashingEmbedder, SemanticCache, SQLiteStore
from semantic_cache.cache import _default_is_cacheable

# ---------------------------------------------------------------------------
# is_cacheable
# ---------------------------------------------------------------------------


class TestIsCacheable:
    def test_default_rejects_empty(self) -> None:
        assert _default_is_cacheable("") is False

    def test_default_rejects_whitespace(self) -> None:
        assert _default_is_cacheable("   ") is False
        assert _default_is_cacheable("\t\n") is False

    def test_default_rejects_short(self) -> None:
        assert _default_is_cacheable("a") is False
        assert _default_is_cacheable("ab") is False

    def test_default_accepts_three_chars(self) -> None:
        assert _default_is_cacheable("abc") is True

    def test_default_accepts_longer(self) -> None:
        assert _default_is_cacheable("hello world") is True

    def test_default_rejects_non_string(self) -> None:
        assert _default_is_cacheable(None) is False  # type: ignore[arg-type]
        assert _default_is_cacheable(42) is False  # type: ignore[arg-type]
        assert _default_is_cacheable([]) is False  # type: ignore[arg-type]

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
            return "ab"  # too short

        result = cache.get_or_compute("hello world", compute)
        assert result.hit is False
        assert result.response == "ab"
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
    def test_dimensions_set_in_init(self) -> None:
        """OllamaEmbedder must set self.dimensions in __init__ (thread-safe)."""
        from semantic_cache.embedders import OllamaEmbedder

        # We can't actually connect to Ollama in CI, so we test that
        # _probe_dimensions returns a sane fallback when Ollama is unreachable.
        embedder = OllamaEmbedder(
            model_name="nomic-embed-text",
            base_url="http://127.0.0.1:1",  # closed port → connection refused
            timeout=0.5,
        )
        assert embedder.dimensions is not None
        assert isinstance(embedder.dimensions, int)
        assert embedder.dimensions > 0

    def test_explicit_dimensions(self) -> None:
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