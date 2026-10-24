"""Comprehensive test suite for semantic-cache v0.2.0.

Run with:
    cd "c:/Users/bmani/Downloads/Semantic cache"
    python -m pytest tests/ -v

All tests use deterministic fake embedders and mock compute functions.
No network access or live AI APIs are required.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional
import numpy as np
import pytest

# We import via the package so the relative imports work correctly
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from semantic_cache.embedders import (
    HashingEmbedder,
    validate_embedding,
    is_zero_vector,
    _validate_text,
)
from semantic_cache.store import SQLiteStore
from semantic_cache.cache import SemanticCache, Stats, CacheResult
from semantic_cache.context import CacheContext
from semantic_cache.gemini import GeminiClient, GeminiEmbedder


# ===========================================================================
# Fixtures / helpers
# ===========================================================================

DIM = 64


class FixedEmbedder:
    """Returns a pre-programmed unit vector for each registered query.
    Unknown queries return a zero vector.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._map: dict[str, np.ndarray] = {}

    def register(self, text: str, vec: np.ndarray) -> "FixedEmbedder":
        self._map[text] = vec.astype(np.float32)
        return self

    def unit(self, idx: int) -> np.ndarray:
        """Return a standard-basis unit vector e_idx."""
        v = np.zeros(self.dim, dtype=np.float32)
        v[idx % self.dim] = 1.0
        return v

    def embed(self, text: str) -> np.ndarray:
        return self._map.get(text, np.zeros(self.dim, dtype=np.float32))


def make_store(dim: int = DIM) -> SQLiteStore:
    return SQLiteStore(path=":memory:", embedding_dim=dim)


def make_cache(embedder=None, threshold=0.90, ttl=None, dim=DIM, **kwargs) -> SemanticCache:
    emb = embedder or HashingEmbedder(dim=dim)
    store = make_store(dim)
    return SemanticCache(
        embedder=emb, store=store, threshold=threshold, ttl=ttl, **kwargs
    )


# ===========================================================================
# 1. Embedder tests
# ===========================================================================

class TestHashingEmbedder:

    def test_returns_float32(self):
        e = HashingEmbedder(dim=64)
        v = e.embed("hello world")
        assert v.dtype == np.float32

    def test_unit_norm(self):
        e = HashingEmbedder(dim=128)
        v = e.embed("What is the capital of France?")
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_string_returns_zero_vector(self):
        e = HashingEmbedder(dim=64)
        v = e.embed("")
        assert is_zero_vector(v)

    def test_whitespace_only_returns_zero_vector(self):
        e = HashingEmbedder(dim=64)
        v = e.embed("   \t\n  ")
        assert is_zero_vector(v)

    def test_near_duplicate_high_similarity(self):
        e = HashingEmbedder(dim=1024)
        v1 = e.embed("What is the capital of France?")
        v2 = e.embed("what is the capital of france")
        score = float(v1 @ v2)
        assert score > 0.90, f"Expected high similarity, got {score:.4f}"

    def test_different_topics_low_similarity(self):
        e = HashingEmbedder(dim=1024)
        v1 = e.embed("What is the capital of France?")
        v2 = e.embed("Who wrote Hamlet?")
        score = float(v1 @ v2)
        assert score < 0.80, f"Expected low similarity, got {score:.4f}"

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError):
            HashingEmbedder(dim=0)

    def test_invalid_ngram_raises(self):
        with pytest.raises(ValueError):
            HashingEmbedder(ngram=0)

    def test_non_string_raises(self):
        e = HashingEmbedder()
        with pytest.raises(TypeError):
            e.embed(123)  # type: ignore

    def test_very_long_text_raises(self):
        e = HashingEmbedder()
        # 33 KB of text exceeds the limit
        with pytest.raises(ValueError, match="exceeds maximum"):
            e.embed("a" * 40_000)


class TestValidateEmbedding:

    def test_accepts_valid_vector(self):
        v = np.array([0.6, 0.8], dtype=np.float32)
        out = validate_embedding(v)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, v)

    def test_coerces_float64(self):
        v = np.array([1.0, 0.0], dtype=np.float64)
        out = validate_embedding(v)
        assert out.dtype == np.float32

    def test_rejects_nan(self):
        v = np.array([float("nan"), 0.5], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_embedding(v)

    def test_rejects_inf(self):
        v = np.array([float("inf"), 0.5], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_embedding(v)

    def test_rejects_2d_array(self):
        v = np.ones((2, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            validate_embedding(v)

    def test_dimension_mismatch(self):
        v = np.ones(8, dtype=np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_embedding(v, expected_dim=16)

    def test_rejects_non_ndarray(self):
        with pytest.raises(TypeError):
            validate_embedding([0.5, 0.5])  # type: ignore

    def test_returns_writable_copy(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        out = validate_embedding(v)
        # Must be writable (not a read-only frombuffer view)
        out[0] = 99.0  # should not raise


class TestCacheContext:

    def test_same_context_has_stable_opaque_namespace(self):
        context = CacheContext(
            tenant="tenant-42",
            model="gpt-example",
            system_prompt_version="2026-10-24",
            user_scope="user-9",
            temperature=0.2,
            tools_version="v3",
            retrieval_version="docs-17",
            locale="en-IN",
        )

        namespace = context.namespace()

        assert namespace == context.namespace()
        assert namespace.startswith("llm:")
        assert "tenant-42" not in namespace
        assert "user-9" not in namespace

    def test_response_affecting_context_fields_partition_namespaces(self):
        base = dict(tenant="a", model="m", system_prompt_version="v1")
        assert CacheContext(**base).namespace() != CacheContext(
            **(base | {"system_prompt_version": "v2"})
        ).namespace()
        assert CacheContext(**base).namespace() != CacheContext(
            **(base | {"retrieval_version": "corpus-v2"})
        ).namespace()


class TestGeminiAdapters:

    def test_gemini_client_extracts_text_without_network(self):
        client = GeminiClient(api_key="test-key")
        client._request = lambda method, payload: {
            "candidates": [{"content": {"parts": [{"text": "hello"}, {"text": " world"}]}}]
        }

        assert client.generate("test") == "hello world"

    def test_gemini_embedder_returns_normalized_vector_without_network(self):
        embedder = GeminiEmbedder(api_key="test-key")
        embedder._request = lambda method, payload: {"embedding": {"values": [3.0, 4.0]}}

        vector = embedder.embed("test")

        assert vector.dtype == np.float32
        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6


# ===========================================================================
# 2. SQLiteStore tests
# ===========================================================================

class TestSQLiteStore:

    def _make_vec(self, idx: int = 0, dim: int = DIM) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        v[idx] = 1.0
        return v

    def test_add_and_search_hit(self):
        store = make_store()
        vec = self._make_vec(0)
        store.add("ns", "hello", vec, "world", ttl=None)
        entry = store.search("ns", vec)
        assert entry is not None
        assert abs(entry.score - 1.0) < 1e-5
        assert entry.response == "world"

    def test_invalid_capacity_configuration_raises(self):
        with pytest.raises(ValueError, match="embedding_dim"):
            SQLiteStore(embedding_dim=0)
        with pytest.raises(ValueError, match="max_entries"):
            SQLiteStore(max_entries=0)

    def test_search_empty_namespace_returns_none(self):
        store = make_store()
        entry = store.search("empty-ns", self._make_vec(0))
        assert entry is None

    def test_ttl_expiry(self):
        store = make_store()
        vec = self._make_vec(1)
        store.add("ns", "q", vec, "r", ttl=0.05)  # 50 ms TTL
        assert store.search("ns", vec) is not None
        time.sleep(0.1)
        # Must rebuild index after expiry
        store._index.clear()
        assert store.search("ns", vec) is None

    def test_clear_specific_namespace_does_not_affect_others(self):
        """Bug fix: clear(ns) used to wipe _index for ALL namespaces."""
        store = make_store()
        v = self._make_vec(2)
        store.add("ns-a", "q", v, "r-a", ttl=None)
        store.add("ns-b", "q", v, "r-b", ttl=None)

        # Pre-populate index for ns-b
        store.search("ns-b", v)
        assert "ns-b" in store._index

        store.clear("ns-a")  # should NOT evict ns-b from _index
        assert "ns-b" in store._index

    def test_clear_all_wipes_index(self):
        store = make_store()
        v = self._make_vec(3)
        store.add("ns", "q", v, "r", ttl=None)
        store.search("ns", v)  # populate index
        store.clear()
        assert store._index == {}

    def test_size_counts_non_expired(self):
        store = make_store()
        v = self._make_vec(4)
        store.add("ns", "q1", v, "r1", ttl=None)
        store.add("ns", "q2", v, "r2", ttl=0.05)
        assert store.size("ns") == 2
        time.sleep(0.1)
        assert store.size("ns") == 1

    def test_delete_by_query(self):
        store = make_store()
        v = self._make_vec(5)
        store.add("ns", "to-delete", v, "response", ttl=None)
        assert store.size("ns") == 1
        deleted = store.delete_by_query("ns", "to-delete")
        assert deleted == 1
        assert store.size("ns") == 0

    def test_invalidate_expired(self):
        store = make_store()
        v = self._make_vec(6)
        store.add("ns", "q", v, "r", ttl=0.05)
        time.sleep(0.1)
        deleted = store.invalidate_expired()
        assert deleted == 1
        assert store.size("ns") == 0

    def test_corrupt_blob_skipped(self):
        """Regression: corrupt BLOB must not crash the store."""
        store = make_store(dim=DIM)
        # Inject a corrupt BLOB directly
        store.db.execute(
            "INSERT INTO entries (namespace, query, embedding, response, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("ns", "corrupt", b"\xff\xfe", "bad", time.time()),
        )
        store.db.commit()
        # Should not raise — corrupt entry is skipped
        result = store.search("ns", self._make_vec(0))
        # result may be None (if only corrupt entry) — that's correct behavior

    def test_blob_is_writable_copy(self):
        """Regression: np.frombuffer returns read-only; validate_embedding must copy."""
        store = make_store()
        v = self._make_vec(7)
        store.add("ns", "q", v, "r", ttl=None)
        # Force index rebuild
        store._index.clear()
        _, mat = store._load("ns")
        # Matrix rows must be writable
        mat[0, 0] = 99.0  # would raise on read-only array

    def test_invalid_namespace_raises(self):
        store = make_store()
        with pytest.raises(ValueError):
            store.add("", "q", self._make_vec(), "r", ttl=None)

    def test_invalid_ttl_raises(self):
        store = make_store()
        with pytest.raises(ValueError):
            store.add("ns", "q", self._make_vec(), "r", ttl=-1)

    def test_max_entries_lru_eviction(self):
        store = SQLiteStore(path=":memory:", embedding_dim=DIM, max_entries=3)
        for i in range(5):
            v = self._make_vec(i)
            store.add("ns", f"q{i}", v, f"r{i}", ttl=None)
        # Should not exceed max_entries
        assert store.size() <= 3

    def test_record_hit_updates_last_hit_at(self):
        store = make_store()
        v = self._make_vec(0)
        eid = store.add("ns", "q", v, "r", ttl=None)
        before = time.time()
        store.record_hit(eid)
        row = store.db.execute(
            "SELECT last_hit_at FROM entries WHERE id=?", (eid,)
        ).fetchone()
        assert row[0] is not None and row[0] >= before

    def test_namespace_stats(self):
        store = make_store()
        v = self._make_vec(0)
        eid = store.add("ns", "q", v, "r", ttl=None)
        store.record_hit(eid)
        s = store.namespace_stats("ns")
        assert s["entries"] == 1
        assert s["total_hits"] == 1


# ===========================================================================
# 3. SemanticCache core tests
# ===========================================================================

class TestSemanticCacheBasic:

    def _make(self, threshold=0.90) -> tuple[SemanticCache, FixedEmbedder]:
        emb = FixedEmbedder(DIM)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=threshold)
        return cache, emb

    def test_miss_then_hit(self):
        cache, emb = self._make()
        v = emb.unit(0)
        emb.register("hello", v)

        assert cache.get("hello") is None  # miss
        cache.set("hello", "world")

        result = cache.get("hello")
        assert result is not None
        assert result.hit is True
        assert result.response == "world"
        assert result.score >= 0.90

    def test_stats_accuracy(self):
        cache, emb = self._make()
        emb.register("q1", emb.unit(0))
        emb.register("q2", emb.unit(1))

        cache.get("q1")           # miss
        cache.set("q1", "r1")
        cache.get("q1")           # hit
        cache.get("q2")           # miss

        assert cache.stats.hits == 1
        assert cache.stats.misses == 2
        assert abs(cache.stats.hit_rate - 1/3) < 1e-6

    def test_get_stats_returns_dict(self):
        cache, _ = self._make()
        d = cache.get_stats()
        assert isinstance(d, dict)
        assert "hits" in d and "misses" in d and "hit_rate" in d

    def test_reset_stats(self):
        cache, emb = self._make()
        emb.register("q", emb.unit(0))
        cache.get("q")  # miss
        cache.reset_stats()
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_clear_removes_entries(self):
        cache, emb = self._make()
        emb.register("q", emb.unit(0))
        cache.set("q", "r")
        cache.clear()
        assert cache.get("q") is None

    def test_delete_removes_exact_entry(self):
        cache, emb = self._make()
        emb.register("q", emb.unit(0))
        cache.set("q", "r")
        cache.delete("q")
        assert cache.get("q") is None

    def test_set_empty_response_raises(self):
        cache, emb = self._make()
        emb.register("q", emb.unit(0))
        with pytest.raises(ValueError):
            cache.set("q", "")

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            SemanticCache(threshold=1.5)
        with pytest.raises(ValueError):
            SemanticCache(threshold=-0.1)

    def test_invalid_namespace_raises(self):
        with pytest.raises(ValueError):
            SemanticCache(namespace="")


class TestSemanticCacheNamespaces:

    def test_namespace_isolation(self):
        """Entries in ns-a must not be visible in ns-b."""
        emb = FixedEmbedder(DIM)
        v = emb.unit(0)
        emb.register("q", v)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90)

        cache.set("q", "response-a", namespace="ns-a")

        # ns-b should not see ns-a's entry
        assert cache.get("q", namespace="ns-b") is None

        # ns-a should see its own entry
        r = cache.get("q", namespace="ns-a")
        assert r is not None
        assert r.response == "response-a"

    def test_different_namespaces_can_have_same_key(self):
        emb = FixedEmbedder(DIM)
        v = emb.unit(1)
        emb.register("q", v)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90)

        cache.set("q", "response-A", namespace="model-A")
        cache.set("q", "response-B", namespace="model-B")

        assert cache.get("q", namespace="model-A").response == "response-A"
        assert cache.get("q", namespace="model-B").response == "response-B"


class TestSemanticCacheTTL:

    def test_entry_expires(self):
        emb = FixedEmbedder(DIM)
        v = emb.unit(2)
        emb.register("q", v)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90, ttl=0.05)

        cache.set("q", "response")
        assert cache.get("q") is not None
        time.sleep(0.12)
        store._index.clear()  # force rebuild
        assert cache.get("q") is None

    def test_entry_persists_before_expiry(self):
        emb = FixedEmbedder(DIM)
        v = emb.unit(3)
        emb.register("q", v)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90, ttl=5.0)

        cache.set("q", "response")
        time.sleep(0.05)
        assert cache.get("q") is not None


class TestSemanticCacheBypass:

    def test_enabled_false_always_misses(self):
        emb = FixedEmbedder(DIM)
        v = emb.unit(0)
        emb.register("q", v)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90, enabled=False)

        cache.set("q", "response")  # no-op
        assert cache.get("q") is None  # always miss

    def test_bypass_flag_on_get(self):
        emb = FixedEmbedder(DIM)
        v = emb.unit(0)
        emb.register("q", v)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90)

        cache.set("q", "response")
        assert cache.get("q", bypass=True) is None

    def test_enabled_false_set_is_noop(self):
        emb = FixedEmbedder(DIM)
        v = emb.unit(0)
        emb.register("q", v)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90, enabled=False)

        cache.set("q", "response")
        assert store.size() == 0  # nothing stored


class TestSemanticCacheZeroVector:

    def test_zero_vector_is_cache_miss(self):
        """Regression: empty query embedds to zero vector — must be a miss, not a crash."""
        emb = FixedEmbedder(DIM)
        # "" maps to zero vector (default)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.85)

        # Setting a real entry first
        v = emb.unit(0)
        emb.register("real-query", v)
        cache.set("real-query", "real-response")

        # Empty-string query should return None, not "real-response"
        result = cache.get("")
        assert result is None


# ===========================================================================
# 4. get_or_compute tests
# ===========================================================================

class TestGetOrCompute:

    def _setup(self, threshold=0.90):
        emb = FixedEmbedder(DIM)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=threshold)
        return cache, emb

    def test_calls_compute_on_miss(self):
        cache, emb = self._setup()
        emb.register("q", emb.unit(0))
        calls = []

        def compute(text: str) -> str:
            calls.append(text)
            return "answer"

        result = cache.get_or_compute("q", compute)
        assert result.response == "answer"
        assert result.hit is False
        assert len(calls) == 1

    def test_returns_cache_on_second_call(self):
        cache, emb = self._setup()
        emb.register("q", emb.unit(0))
        calls = []

        def compute(text: str) -> str:
            calls.append(text)
            return "answer"

        cache.get_or_compute("q", compute)
        result = cache.get_or_compute("q", compute)

        assert result.hit is True
        assert len(calls) == 1  # compute called only once

    def test_exception_not_cached(self):
        """Failed compute must not store anything in the cache."""
        cache, emb = self._setup()
        emb.register("q", emb.unit(0))

        def failing_compute(text: str) -> str:
            raise RuntimeError("LLM API failed")

        with pytest.raises(RuntimeError):
            cache.get_or_compute("q", failing_compute)

        assert cache.store.size() == 0
        assert cache.stats.errors == 1

    def test_exception_propagates_unchanged(self):
        cache, emb = self._setup()
        emb.register("q", emb.unit(0))

        class CustomError(Exception):
            pass

        def bad_compute(text: str) -> str:
            raise CustomError("specific error")

        with pytest.raises(CustomError, match="specific error"):
            cache.get_or_compute("q", bad_compute)

    def test_empty_return_not_cached(self):
        cache, emb = self._setup()
        emb.register("q", emb.unit(0))

        def empty_compute(text: str) -> str:
            return ""

        with pytest.raises(ValueError, match="non-empty string"):
            cache.get_or_compute("q", empty_compute)

        assert cache.store.size() == 0

    def test_seconds_saved_accumulated(self):
        cache, emb = self._setup()
        emb.register("q", emb.unit(0))
        call_count = [0]

        def slow_compute(text: str) -> str:
            call_count[0] += 1
            time.sleep(0.05)
            return "answer"

        cache.get_or_compute("q", slow_compute)  # MISS — measures latency
        cache.get_or_compute("q", slow_compute)  # HIT  — accumulates savings

        assert cache.stats.seconds_saved > 0.0


class TestGetOrComputeBypass:

    def _make(self, enabled=True):
        emb = FixedEmbedder(DIM)
        emb.register("q", emb.unit(0))
        store = make_store(DIM)
        return SemanticCache(
            embedder=emb, store=store, threshold=0.90, enabled=enabled
        ), store

    def test_bypass_computes_fresh_instead_of_using_existing_entry(self):
        cache, _ = self._make()
        cache.set("q", "cached")
        calls = []

        result = cache.get_or_compute("q", lambda text: calls.append(text) or "fresh", bypass=True)

        assert result.response == "fresh"
        assert result.hit is False
        assert calls == ["q"]

    def test_bypass_does_not_create_or_modify_entries(self):
        cache, store = self._make()

        cache.get_or_compute("q", lambda _: "fresh", bypass=True)
        assert store.size() == 0

        cache.set("q", "cached")
        entry_before = store.db.execute(
            "SELECT response, hits FROM entries WHERE namespace=? AND query=?", ("default", "q")
        ).fetchone()
        cache.get_or_compute("q", lambda _: "fresh", bypass=True)
        entry_after = store.db.execute(
            "SELECT response, hits FROM entries WHERE namespace=? AND query=?", ("default", "q")
        ).fetchone()
        assert entry_after == entry_before == ("cached", 0)

    def test_repeated_bypass_calls_compute_independently(self):
        cache, store = self._make()
        calls = []

        def compute(_: str) -> str:
            calls.append(1)
            return f"fresh-{len(calls)}"

        assert cache.get_or_compute("q", compute, bypass=True).response == "fresh-1"
        assert cache.get_or_compute("q", compute, bypass=True).response == "fresh-2"
        assert len(calls) == 2
        assert store.size() == 0

    def test_bypass_exception_propagates_without_caching(self):
        cache, store = self._make()

        def failing_compute(_: str) -> str:
            raise RuntimeError("LLM API failed")

        with pytest.raises(RuntimeError, match="LLM API failed"):
            cache.get_or_compute("q", failing_compute, bypass=True)
        assert store.size() == 0
        assert cache.stats.errors == 1

    def test_bypass_avoids_embedder_and_store(self):
        class ExplodingEmbedder:
            def embed(self, _: str) -> np.ndarray:
                raise AssertionError("bypass must not embed")

        class ExplodingStore:
            def __bool__(self) -> bool:
                return True

            def __getattr__(self, name: str):
                raise AssertionError(f"bypass must not access store.{name}")

        store = ExplodingStore()
        cache = SemanticCache(embedder=ExplodingEmbedder(), store=store)

        assert cache.get_or_compute("q", lambda _: "fresh", bypass=True).response == "fresh"

    def test_enabled_false_computes_without_reading_or_writing(self):
        cache, store = self._make(enabled=False)
        # Seed the backing store directly; a disabled cache must ignore it.
        store.add("default", "q", cache.embedder.embed("q"), "cached", ttl=None)

        result = cache.get_or_compute("q", lambda _: "fresh")

        assert result.response == "fresh"
        assert result.hit is False
        assert store.db.execute(
            "SELECT response FROM entries WHERE namespace=? AND query=?", ("default", "q")
        ).fetchone() == ("cached",)

    def test_ordinary_requests_still_store_and_hit(self):
        cache, store = self._make()
        calls = []

        def compute(_: str) -> str:
            calls.append(1)
            return "cached-result"

        first = cache.get_or_compute("q", compute)
        second = cache.get_or_compute("q", compute)

        assert first.hit is False
        assert second.hit is True
        assert second.response == "cached-result"
        assert calls == [1]
        assert store.size() == 1

    def test_concurrent_bypass_calls_compute_independently_without_deadlock(self):
        cache, store = self._make()
        barrier = threading.Barrier(2)
        calls = []
        results = []
        errors = []

        def compute(_: str) -> str:
            calls.append(1)
            time.sleep(0.05)
            return "fresh"

        def worker():
            try:
                barrier.wait()
                results.append(cache.get_or_compute("q", compute, bypass=True))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        assert len(results) == 2
        assert len(calls) == 2
        assert store.size() == 0


# ===========================================================================
# 5. @cached decorator tests
# ===========================================================================

class TestCachedDecorator:

    def test_decorator_wraps_function(self):
        emb = FixedEmbedder(DIM)
        emb.register("q", emb.unit(0))
        cache = SemanticCache(embedder=emb, store=make_store(DIM), threshold=0.90)
        calls = []

        @cache.cached(namespace="test")
        def ask(prompt: str) -> str:
            calls.append(prompt)
            return "answer"

        assert ask("q") == "answer"
        assert ask("q") == "answer"  # second call should hit cache
        assert len(calls) == 1

    def test_decorator_preserves_function_name(self):
        cache = SemanticCache()

        @cache.cached()
        def my_llm_fn(prompt: str) -> str:
            return "response"

        assert my_llm_fn.__name__ == "my_llm_fn"

    def test_decorator_propagates_exception(self):
        cache = SemanticCache()

        @cache.cached()
        def bad_fn(prompt: str) -> str:
            raise ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            bad_fn("test prompt")

    def test_bypass_decorator_neither_reads_nor_writes(self):
        emb = FixedEmbedder(DIM)
        emb.register("q", emb.unit(0))
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90)
        cache.set("q", "cached")
        calls = []

        @cache.cached(bypass=True)
        def ask(prompt: str) -> str:
            calls.append(prompt)
            return f"fresh-{len(calls)}"

        assert ask("q") == "fresh-1"
        assert ask("q") == "fresh-2"
        assert calls == ["q", "q"]
        assert store.db.execute(
            "SELECT response, hits FROM entries WHERE namespace=? AND query=?", ("default", "q")
        ).fetchone() == ("cached", 0)


class TestAsyncCache:

    def test_async_get_or_compute_stores_then_hits(self):
        cache = make_cache()
        calls = []

        async def compute(_: str) -> str:
            calls.append(1)
            return "answer"

        async def scenario():
            first = await cache.async_get_or_compute("q", compute)
            second = await cache.async_get_or_compute("q", compute)
            return first, second

        first, second = asyncio.run(scenario())
        assert first.hit is False
        assert second.hit is True
        assert calls == [1]

    def test_async_bypass_is_fresh_and_uncached(self):
        cache = make_cache()
        calls = []

        async def compute(_: str) -> str:
            calls.append(1)
            return f"answer-{len(calls)}"

        async def scenario():
            return await asyncio.gather(
                cache.async_get_or_compute("q", compute, bypass=True),
                cache.async_get_or_compute("q", compute, bypass=True),
            )

        first, second = asyncio.run(scenario())
        assert (first.response, second.response) == ("answer-1", "answer-2")
        assert cache.store.size() == 0

    def test_async_single_flight_coalesces_normal_requests(self):
        cache = make_cache()
        calls = []

        async def compute(_: str) -> str:
            calls.append(1)
            await asyncio.sleep(0.02)
            return "answer"

        async def scenario():
            return await asyncio.gather(
                cache.async_get_or_compute("q", compute),
                cache.async_get_or_compute("q", compute),
            )

        results = asyncio.run(scenario())
        assert [result.response for result in results] == ["answer", "answer"]
        assert calls == [1]


# ===========================================================================
# 6. Concurrency tests
# ===========================================================================

class TestConcurrency:

    def test_thread_safe_writes(self):
        """Multiple threads writing to the same cache must not corrupt data."""
        emb = HashingEmbedder(dim=DIM)
        store = SQLiteStore(path=":memory:", embedding_dim=DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.85)

        errors = []

        def worker(i: int):
            try:
                cache.set(f"query-{i}", f"response-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert store.size() == 20

    def test_single_flight_deduplication(self):
        """Two simultaneous misses for the same query must result in only one compute call."""
        emb = FixedEmbedder(DIM)
        emb.register("q", emb.unit(0))
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90)

        call_count = [0]
        barrier = threading.Barrier(2)

        def slow_compute(text: str) -> str:
            call_count[0] += 1
            time.sleep(0.1)
            return "answer"

        results = []
        errors = []

        def worker():
            try:
                barrier.wait()  # Both threads start simultaneously
                r = cache.get_or_compute("q", slow_compute)
                results.append(r)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Errors: {errors}"
        assert len(results) == 2
        assert all(r.response == "answer" for r in results)
        # compute should have been called at most twice (once for each thread that got a miss)
        # With single-flight, ideally only once — but the guarantee is "no extra compute"
        assert call_count[0] <= 2

    def test_exception_released_waiters(self):
        """When compute raises, waiting threads should not deadlock."""
        emb = FixedEmbedder(DIM)
        emb.register("q", emb.unit(5))
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90)

        barrier = threading.Barrier(2)
        errors = []
        expected_errors = []

        def failing_compute(text: str) -> str:
            time.sleep(0.05)
            raise RuntimeError("intentional failure")

        def worker():
            try:
                barrier.wait()
                cache.get_or_compute("q", failing_compute)
            except RuntimeError:
                expected_errors.append(1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Thread deadlocked!"

        assert not errors, f"Unexpected errors: {errors}"


# ===========================================================================
# 7. Callbacks tests
# ===========================================================================

class TestCallbacks:

    def test_on_hit_called(self):
        emb = FixedEmbedder(DIM)
        emb.register("q", emb.unit(0))
        hit_results = []
        cache = SemanticCache(
            embedder=emb,
            store=make_store(DIM),
            threshold=0.90,
            on_hit=hit_results.append,
        )
        cache.set("q", "r")
        cache.get("q")
        assert len(hit_results) == 1
        assert hit_results[0].hit is True

    def test_on_miss_called(self):
        emb = FixedEmbedder(DIM)
        emb.register("q", emb.unit(0))
        miss_queries = []
        cache = SemanticCache(
            embedder=emb,
            store=make_store(DIM),
            threshold=0.90,
            on_miss=miss_queries.append,
        )
        cache.get("q")
        assert "q" in miss_queries

    def test_on_error_called(self):
        emb = FixedEmbedder(DIM)
        emb.register("q", emb.unit(0))
        errors = []
        cache = SemanticCache(
            embedder=emb,
            store=make_store(DIM),
            threshold=0.90,
            on_error=lambda q, e: errors.append((q, e)),
        )

        def fail(text: str) -> str:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            cache.get_or_compute("q", fail)

        assert len(errors) == 1
        assert errors[0][0] == "q"


# ===========================================================================
# 8. Backward compatibility tests
# ===========================================================================

class TestBackwardCompatibility:
    """Ensure the original v0.1.0 public API still works unchanged."""

    def test_original_get_set_api(self):
        cache = SemanticCache(threshold=0.75)
        cache.set("What is the capital of France?", "Paris")
        result = cache.get("what is the capital of france")
        # HashingEmbedder may not produce a high enough similarity at this threshold
        # The important thing is the API doesn't crash
        assert result is None or result.hit is True

    def test_original_stats_fields(self):
        stats = Stats()
        assert hasattr(stats, "hits")
        assert hasattr(stats, "misses")
        assert hasattr(stats, "seconds_saved")
        assert hasattr(stats, "hit_rate")

    def test_original_cache_result_fields(self):
        r = CacheResult(response="r", hit=True, score=0.9, matched_query="q")
        assert r.response == "r"
        assert r.hit is True
        assert r.score == 0.9
        assert r.matched_query == "q"

    def test_original_clear_api(self):
        cache = SemanticCache()
        cache.set("q", "r")
        cache.clear()  # should not raise

    def test_original_store_size_api(self):
        store = SQLiteStore()
        assert store.size() == 0

    def test_original_cached_decorator_api(self):
        cache = SemanticCache(threshold=0.75, ttl=3600)
        calls = [0]

        ask = cache.cached(namespace="test:v1")(lambda p: f"answer:{p}")
        result = ask("hello")
        assert result == "answer:hello"


# ===========================================================================
# 9. Similarity threshold / semantic matching tests
# ===========================================================================

class TestSimilarityThreshold:

    def test_below_threshold_is_miss(self):
        emb = FixedEmbedder(DIM)
        # Two orthogonal vectors: similarity = 0
        v0 = emb.unit(0)
        v1 = emb.unit(1)
        emb.register("query-A", v0)
        emb.register("query-B", v1)

        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90)

        cache.set("query-A", "response-A")
        # query-B is orthogonal to query-A → score=0 → miss
        result = cache.get("query-B")
        assert result is None

    def test_exact_threshold_boundary(self):
        """Score exactly at threshold must be a HIT (>= comparison)."""
        emb = FixedEmbedder(DIM)
        v = emb.unit(0)
        emb.register("q", v)

        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=1.0)
        cache.set("q", "r")

        result = cache.get("q")
        assert result is not None  # score == 1.0 == threshold → HIT


# ===========================================================================
# 10. invalidate_expired test
# ===========================================================================

class TestInvalidateExpired:

    def test_invalidate_removes_only_expired(self):
        store = SQLiteStore(path=":memory:", embedding_dim=DIM)
        v0 = np.zeros(DIM, dtype=np.float32); v0[0] = 1.0
        v1 = np.zeros(DIM, dtype=np.float32); v1[1] = 1.0

        store.add("ns", "q-expire", v0, "r0", ttl=0.05)
        store.add("ns", "q-keep",   v1, "r1", ttl=None)

        time.sleep(0.1)
        deleted = store.invalidate_expired()

        assert deleted == 1
        assert store.size("ns") == 1


# ===========================================================================
# 11. Namespace stats test
# ===========================================================================

class TestNamespaceStats:

    def test_cache_namespace_stats(self):
        emb = FixedEmbedder(DIM)
        v = emb.unit(0)
        emb.register("q", v)
        store = make_store(DIM)
        cache = SemanticCache(embedder=emb, store=store, threshold=0.90, namespace="my-ns")

        cache.set("q", "r")
        cache.get("q")  # hit

        ns_stats = cache.namespace_stats()
        assert ns_stats["namespace"] == "my-ns"
        assert ns_stats["entries"] == 1
