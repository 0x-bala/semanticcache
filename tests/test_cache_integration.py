"""Integration tests for SemanticCache with real ask() workflow and guard.

Tests cover:
a) Initial MISSes that populate cache via mocked LLM
b) Paraphrased HITs with unchanged LLM call count
c) Similar wording queries that must MISS (distinguishing different entities)
d) Edge cases ("", "???", Tamil text, 5000-char string)
e) Multi-threaded concurrency (8 threads x 40 calls)
f) Namespace isolation (model name / prompt version)
g) Persistence across restarts with file-based SQLiteStore
h) TTL expiration behavior
Plus:
- Mocked embedder returning real Gemini scores (0.9816, 0.88-0.91) proving:
  at threshold 0.90 all group-(c) queries MISS and all four paraphrases HIT
- Exact repeats make zero embedding calls
- LRU avoids a second embedding call
- Guard can be disabled (demonstrating false hit without guard)
- Verifier callable integration
- Live Gemini tests using real GeminiClient.
"""
from __future__ import annotations

import concurrent.futures
import os
import sqlite3
import tempfile
import time
from collections.abc import Callable

import numpy as np
import pytest

from semantic_cache import (
    GeminiAPIError,
    GeminiClient,
    GeminiEmbedder,
    HashingEmbedder,
    SemanticCache,
    SQLiteStore,
)

# ---------------------------------------------------------------------------
# Test Prompts & Real Gemini Scores
# ---------------------------------------------------------------------------

SEEDS = [
    ("What is the capital of France?", "Paris"),
    ("How do I reset my password?", "Go to settings -> Security -> Reset Password."),
    ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
    ("Convert 10 km to miles", "10 km is equal to approximately 6.21371 miles."),
    ("Who is the CEO of Tesla?", "Elon Musk is the CEO of Tesla."),
]

GROUP_B_PARAPHRASES = [
    ("what's the capital of france", "What is the capital of France?"),
    ("How can I reset my password?", "How do I reset my password?"),
    ("Who wrote the play Hamlet?", "Who wrote Hamlet?"),
    ("Capital city of France?", "What is the capital of France?"),
]

GROUP_C_DIFFERENT_ENTITIES = [
    ("What is the capital of Spain?", "What is the capital of France?"),
    ("How do I reset my router?", "How do I reset my password?"),
    ("Who wrote Macbeth?", "Who wrote Hamlet?"),
    ("Convert 10 miles to km", "Convert 10 km to miles"),
    ("Who is the CEO of Apple?", "Who is the CEO of Tesla?"),
]

# Exact measured scores from GeminiEmbedder (gemini-embedding-001)
REAL_GEMINI_SCORES: dict[str, tuple[int, float]] = {
    "what's the capital of france": (0, 0.9890),
    "How can I reset my password?": (1, 0.9976),
    "Who wrote the play Hamlet?": (2, 0.9854),
    "Capital city of France?": (0, 0.9768),
    "What is the capital of Spain?": (0, 0.8806),
    "How do I reset my router?": (1, 0.8897),
    "Who wrote Macbeth?": (2, 0.9055),
    "Convert 10 miles to km": (3, 0.9816),
    "Who is the CEO of Apple?": (4, 0.8876),
}


class MockRealisticScoreEmbedder:
    """Mock embedder returning the exact measured Gemini cosine similarity scores."""

    def __init__(self, dim: int = 16, model_name: str = "gemini-embedding-001") -> None:
        self.dim = dim
        self.embed_calls: int = 0
        self.model_name = model_name
        self._seed_indices: dict[str, int] = {
            query: idx for idx, (query, _) in enumerate(SEEDS)
        }
        self._custom_scores: dict[str, tuple[int, float]] = {}

    def set_score(self, text: str, seed_idx: int, score: float) -> None:
        """Override cosine similarity for a specific query relative to a seed."""
        self._custom_scores[text] = (seed_idx, score)

    def embed(self, text: str) -> np.ndarray:
        self.embed_calls += 1
        vec = np.zeros(self.dim, dtype=np.float32)

        if text in self._seed_indices:
            idx = self._seed_indices[text]
            vec[idx] = 1.0
            return vec

        if text in self._custom_scores:
            idx, score = self._custom_scores[text]
            vec[idx] = score
            vec[idx + 5] = float(np.sqrt(max(0.0, 1.0 - score * score)))
            return vec

        if text in REAL_GEMINI_SCORES:
            idx, score = REAL_GEMINI_SCORES[text]
            # Construct unit vector with exact cosine similarity = score with seed vector
            vec[idx] = score
            vec[idx + 5] = float(np.sqrt(max(0.0, 1.0 - score * score)))
            return vec

        # Default fallback vector
        vec[10] = 1.0
        return vec


def make_counting_llm() -> tuple[Callable[[str], str], list[int]]:
    calls = [0]

    def llm(prompt: str) -> str:
        calls[0] += 1
        return f"[LLM answer to: {prompt}]"

    return llm, calls


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestCacheIntegration:
    """End-to-end integration test suite using ask() interface with guard and speed tiers."""

    def test_a_misses_call_llm_once(self) -> None:
        """Requirement (a): MISS then LLM called once for each of the 5 queries."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, threshold=0.90)
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="test-group-a")(llm)

        for query, _ in SEEDS:
            res = ask(query)
            assert f"[LLM answer to: {query}]" in res

        assert calls[0] == len(SEEDS), f"Expected {len(SEEDS)} LLM calls, got {calls[0]}"

    def test_b_hits_with_llm_count_unchanged(self) -> None:
        """Requirement (b): HIT with LLM call count unchanged for paraphrases."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, threshold=0.90)
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="test-group-b")(llm)

        # Populate seeds
        for query, _ in SEEDS:
            ask(query)
        assert calls[0] == 5

        # Paraphrases should hit
        for query, ref in GROUP_B_PARAPHRASES:
            res = ask(query)
            assert f"[LLM answer to: {ref}]" in res

        # LLM count must remain 5
        assert calls[0] == 5, f"Expected 5 LLM calls (hits only), got {calls[0]}"

    def test_c_must_be_miss_at_threshold_90_with_guard(self) -> None:
        """Requirement (c): At threshold 0.90, all group (c) queries MISS with guard."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, threshold=0.90, guard=True)
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="test-group-c")(llm)

        # Populate seeds
        for query, _ in SEEDS:
            ask(query)
        assert calls[0] == 5

        # All different entity/inversion queries must MISS and invoke LLM
        for query, _ in GROUP_C_DIFFERENT_ENTITIES:
            ask(query)

        # 5 seeds + 5 misses = 10 total calls
        assert calls[0] == 10, f"Expected 10 total calls, got {calls[0]}"
        assert cache.stats.guard_rejects >= 1  # Unit inversion rejected

    def test_guard_disabled_allows_false_hits(self) -> None:
        """Disabling guard permits false positive hits on high-scoring inversions."""
        embedder = MockRealisticScoreEmbedder()
        # With guard=False, the 0.9816 unit inversion score triggers a false hit
        cache = SemanticCache(embedder=embedder, threshold=0.90, guard=False)
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="test-unguarded")(llm)

        for query, _ in SEEDS:
            ask(query)
        assert calls[0] == 5

        # "Convert 10 miles to km" matches "Convert 10 km to miles" with score 0.9816 >= 0.90
        res = ask("Convert 10 miles to km")
        # Demonstrates false hit on reversed units when guard is disabled
        assert "[LLM answer to: Convert 10 km to miles]" in res
        assert calls[0] == 5  # No LLM call made because of false cache hit!

    def test_verifier_accepts_verify_verdict(self) -> None:
        """A verifier returning True allows borderline VERIFY pairs to hit."""
        embedder = MockRealisticScoreEmbedder()

        # Verifier that approves Macbeth ~ Hamlet
        def custom_verifier(new_q: str, cached_q: str) -> bool:
            return "macbeth" in new_q.lower() and "hamlet" in cached_q.lower()

        cache = SemanticCache(
            embedder=embedder,
            threshold=0.90,
            guard=True,
            verifier=custom_verifier,
        )
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="test-verifier")(llm)

        for query, _ in SEEDS:
            ask(query)
        assert calls[0] == 5

        # "Who wrote Macbeth?" (score 0.9055 within buffer) triggers VERIFY, approved by verifier
        res = ask("Who wrote Macbeth?")
        assert "[LLM answer to: Who wrote Hamlet?]" in res
        assert calls[0] == 5  # HIT accepted by verifier!
        assert cache.stats.verify_calls == 1
        assert cache.stats.verify_accepts == 1

    def test_exact_repeats_make_zero_embedding_calls(self) -> None:
        """Exact repeat queries return immediately via exact-match tier without embedding."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, exact_match=True)
        llm, _ = make_counting_llm()
        ask = cache.cached(namespace="test-exact")(llm)

        query = "What is the capital of France?"
        # First call -> MISS -> 1 embed call
        ask(query)
        assert embedder.embed_calls == 1

        # Second call -> Exact HIT -> 0 additional embed calls!
        res = ask(query)
        assert "[LLM answer to: What is the capital of France?]" in res
        assert embedder.embed_calls == 1, (
            f"Expected exactly 1 embed call, got {embedder.embed_calls}"
        )
        assert cache.stats.exact_hits == 1

    def test_lru_avoids_second_embedding_call(self) -> None:
        """LRU cache retains vectors so identical normalized texts are never embedded twice."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, embedding_lru_capacity=100)

        query = "What is the capital of France?"
        cache.set(query, "Paris")
        assert embedder.embed_calls == 1

        # Query in a different namespace (so exact store index does not short-circuit)
        # The LRU cache must supply the embedding vector without re-calling the embedder
        res = cache.get(query, namespace="other-namespace")
        assert res is None  # Other namespace is empty
        assert embedder.embed_calls == 1, (
            f"Expected embedder to not be called again, calls={embedder.embed_calls}"
        )

    def test_d_edge_cases_must_not_raise(self) -> None:
        """Requirement (d): Edge cases must not raise any exceptions."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder)
        llm, _ = make_counting_llm()
        ask = cache.cached(namespace="test-edge-cases")(llm)

        edge_cases = [
            "",
            "???",
            "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "A" * 5000,
        ]

        for query in edge_cases:
            try:
                res = ask(query)
                assert isinstance(res, str)
            except Exception as exc:
                pytest.fail(f"Query {query[:30]!r} raised unexpected exception: {exc}")

    def test_e_concurrency_8_threads_40_calls(self) -> None:
        """Requirement (e): 8 threads x 40 calls on one cache, no exceptions."""
        embedder = HashingEmbedder(dim=1024)
        cache = SemanticCache(
            embedder=embedder,
            store=SQLiteStore(":memory:", embedding_dim=1024),
        )
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="test-concurrency")(llm)

        def worker(thread_id: int) -> None:
            for i in range(40):
                ask(f"concurrent-prompt-{i % 5}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, t) for t in range(8)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        # Single-flight coalescing ensures exactly 5 compute calls occurred
        assert calls[0] == 5
        assert cache.store.size("test-concurrency") == 5

    def test_f_namespace_isolation(self) -> None:
        """Requirement (f): Different model/prompt namespaces never collide."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder)
        calls_v1 = [0]
        calls_v2 = [0]

        ask_v1 = cache.cached(namespace="model-a:v1")(
            lambda p: (calls_v1.__setitem__(0, calls_v1[0] + 1), f"v1:{p}")[1]
        )
        ask_v2 = cache.cached(namespace="model-a:v2")(
            lambda p: (calls_v2.__setitem__(0, calls_v2[0] + 1), f"v2:{p}")[1]
        )

        prompt = "What is the capital of France?"
        r1 = ask_v1(prompt)
        r2 = ask_v2(prompt)

        assert r1 == f"v1:{prompt}"
        assert r2 == f"v2:{prompt}"
        assert calls_v1[0] == 1
        assert calls_v2[0] == 1

    def test_g_restart_test_file_sqlite_store(self) -> None:
        """Requirement (g): File-based SQLiteStore survives restarts and still hits."""
        embedder = MockRealisticScoreEmbedder()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            llm, calls = make_counting_llm()

            # First instance populates
            store1 = SQLiteStore(path=db_path)
            cache1 = SemanticCache(embedder=embedder, store=store1)
            ask1 = cache1.cached(namespace="restart-test")(llm)
            r1 = ask1("Persistent cache query")
            assert calls[0] == 1
            store1.close()

            # Second instance reads
            store2 = SQLiteStore(path=db_path)
            cache2 = SemanticCache(embedder=embedder, store=store2)
            ask2 = cache2.cached(namespace="restart-test")(llm)
            r2 = ask2("Persistent cache query")
            assert calls[0] == 1  # HIT, no new LLM call
            assert r1 == r2
            store2.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_h_ttl_expiry(self) -> None:
        """Requirement (h): Short TTL entry expires and becomes a MISS."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, ttl=0.1)
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="ttl-test")(llm)

        # First call -> MISS
        r1 = ask("TTL test query")
        assert calls[0] == 1

        # Immediate call -> HIT
        r2 = ask("TTL test query")
        assert calls[0] == 1
        assert r1 == r2

        # Expire
        time.sleep(0.2)

        # Subsequent call -> MISS
        r3 = ask("TTL test query")
        assert calls[0] == 2
        assert r1 == r3

    def test_entity_swap_high_scores_miss_without_verifier(self) -> None:
        """Entity swaps scoring 0.95 and 0.98 must still MISS when verify_always=True."""
        embedder = MockRealisticScoreEmbedder()
        # Seed 0: "What is the capital of France?"
        # Seed 4: "Who is the CEO of Tesla?"
        embedder.set_score("What is the capital of Spain?", seed_idx=0, score=0.98)
        embedder.set_score("Who is the CEO of Apple?", seed_idx=4, score=0.95)

        cache = SemanticCache(
            embedder=embedder,
            threshold=0.90,
            verify_always=True,
            verifier=None,
        )
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="high-score-test")(llm)

        for query, _ in SEEDS:
            ask(query)
        assert calls[0] == 5

        # Both entity swaps score >= 0.95 but trigger VERIFY; without verifier they MISS
        res_spain = ask("What is the capital of Spain?")
        assert calls[0] == 6
        assert "[LLM answer to: What is the capital of Spain?]" in res_spain

        res_apple = ask("Who is the CEO of Apple?")
        assert calls[0] == 7
        assert "[LLM answer to: Who is the CEO of Apple?]" in res_apple

    def test_entity_swap_high_scores_hit_with_verifier(self) -> None:
        """Entity swaps scoring 0.95 and 0.98 HIT if verifier returns True."""
        embedder = MockRealisticScoreEmbedder()
        embedder.set_score("What is the capital of Spain?", seed_idx=0, score=0.98)
        embedder.set_score("Who is the CEO of Apple?", seed_idx=4, score=0.95)

        cache = SemanticCache(
            embedder=embedder,
            threshold=0.90,
            verify_always=True,
            verifier=lambda new_q, cached_q: True,  # Verifier approves
        )
        llm, calls = make_counting_llm()
        ask = cache.cached(namespace="high-score-verifier-test")(llm)

        for query, _ in SEEDS:
            ask(query)
        assert calls[0] == 5

        # Both high-scoring swaps hit through verifier approval without calling LLM
        res_spain = ask("What is the capital of Spain?")
        assert calls[0] == 5
        assert "[LLM answer to: What is the capital of France?]" in res_spain

        res_apple = ask("Who is the CEO of Apple?")
        assert calls[0] == 5
        assert "[LLM answer to: Who is the CEO of Tesla?]" in res_apple
        assert cache.stats.verify_calls == 2
        assert cache.stats.verify_accepts == 2

    def test_sqlite_migration_from_old_schema(self) -> None:
        """Opening a SQLite database created with the old schema migrates and backfills."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            old_db_path = f.name

        try:
            # 1. Create a hand-built database with strictly the old schema
            conn = sqlite3.connect(old_db_path)
            conn.execute(
                """CREATE TABLE entries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace   TEXT    NOT NULL,
                    query       TEXT    NOT NULL,
                    embedding   BLOB    NOT NULL,
                    response    TEXT    NOT NULL,
                    created_at  REAL    NOT NULL,
                    expires_at  REAL,
                    hits        INTEGER NOT NULL DEFAULT 0,
                    last_hit_at REAL
                )"""
            )
            # Insert entries into the old schema
            dim = 16
            vec = np.zeros(dim, dtype=np.float32)
            vec[0] = 1.0
            blob = vec.tobytes()
            conn.execute(
                """INSERT INTO entries
                   (namespace, query, embedding, response, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("default", "What is the capital of France?", blob, "Paris", time.time(), None),
            )
            conn.commit()
            conn.close()

            # 2. Open using SQLiteStore; must migrate without crashing
            store = SQLiteStore(path=old_db_path, embedding_dim=dim)

            # Verify columns were added
            cols = {row[1] for row in store.db.execute("PRAGMA table_info(entries)").fetchall()}
            assert "query_hash" in cols
            assert "embedder" in cols

            # Verify query_hash was backfilled
            row = store.db.execute(
                "SELECT query_hash, embedder FROM entries WHERE query=?",
                ("What is the capital of France?",),
            ).fetchone()
            assert row is not None
            assert row[0] is not None and len(row[0]) == 64  # Valid SHA-256 hash

            # Verify exact match tier works on backfilled legacy entry
            exact_entry = store.find_exact("default", row[0])
            assert exact_entry is not None
            assert exact_entry.response == "Paris"

            # Verify search works
            search_entry = store.search("default", vec)
            assert search_entry is not None
            assert search_entry.response == "Paris"
            store.close()
        finally:
            if os.path.exists(old_db_path):
                os.remove(old_db_path)

    def test_exact_hit_respects_ttl_expiry(self) -> None:
        """Exact-match tier respects TTL expiration."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, ttl=0.1, exact_match=True)
        cache.set("What is the capital of France?", "Paris")

        # Immediate exact hit
        hit = cache.get("What is the capital of France?")
        assert hit is not None
        assert hit.response == "Paris"
        assert cache.stats.exact_hits == 1

        # Wait for TTL expiry
        time.sleep(0.15)
        expired = cache.get("What is the capital of France?")
        assert expired is None

    def test_exact_hit_respects_clear(self) -> None:
        """Exact-match tier respects cache.clear()."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, exact_match=True)
        cache.set("Query to clear", "Answer")

        assert cache.get("Query to clear") is not None
        cache.clear()
        assert cache.get("Query to clear") is None

    def test_exact_hit_respects_namespace_separation(self) -> None:
        """Exact-match tier strictly separates namespaces."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, exact_match=True)
        cache.set("Shared Query", "Answer NS1", namespace="ns1")

        assert cache.get("Shared Query", namespace="ns1") is not None
        assert cache.get("Shared Query", namespace="ns2") is None

    def test_exact_hit_respects_embedder_separation(self) -> None:
        """Exact-match tier strictly separates entries by embedder model name."""
        store = SQLiteStore(":memory:")
        embedder_a = MockRealisticScoreEmbedder(model_name="model-alpha")
        embedder_b = MockRealisticScoreEmbedder(model_name="model-beta")

        cache_a = SemanticCache(embedder=embedder_a, store=store, exact_match=True)
        cache_b = SemanticCache(embedder=embedder_b, store=store, exact_match=True)

        cache_a.set("Cross-model query", "Alpha answer")
        hit_a = cache_a.get("Cross-model query")
        assert hit_a is not None
        assert hit_a.response == "Alpha answer"

        hit_b = cache_b.get("Cross-model query")
        assert hit_b is None
        cache_a.close()
        cache_b.close()
        store.close()

    def test_exact_match_preserves_symbols_and_does_not_cross_hit(self) -> None:
        """Exact-match tier must preserve symbols and never cross-hit on symbol differences."""
        # Embedder that returns orthogonal vectors so semantic search will miss
        embedder = HashingEmbedder(dim=128)
        cache = SemanticCache(embedder=embedder, threshold=0.99, exact_match=True)

        symbol_pairs = [
            ("What is C++?", ["What is C#?", "What is C?"]),
            ("Is x>5?", ["Is x<5?", "Is x=5?"]),
            ("Discount is 5%", ["Discount is $5"]),
            ("Version 3.5", ["Version 35"]),
            ("+91 98765", ["91 98765"]),
        ]

        for stored_query, non_matching_queries in symbol_pairs:
            cache.set(stored_query, f"Response for {stored_query}")
            for non_match in non_matching_queries:
                # Exact match must not hit
                hit = cache.get(non_match)
                assert hit is None, (
                    f"Query {non_match!r} incorrectly hit stored query {stored_query!r}"
                )

        # But casing and trailing question marks still exact-match
        cache.set("What is the capital of France?", "Paris")
        hit = cache.get("what is the capital of france")
        assert hit is not None
        assert hit.response == "Paris"
        assert hit.matched_query == "What is the capital of France?"


    def test_shadow_mode_estimated_false_hit_rate(self) -> None:
        """Shadow mode runs real compute on cache hits and calculates false-hit rate."""
        embedder = MockRealisticScoreEmbedder()
        cache = SemanticCache(embedder=embedder, shadow_rate=1.0)

        # Function that changes answer after the first call
        call_count = [0]

        def dynamic_llm(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "Original Answer"
            return f"Different Answer {call_count[0]}"

        # 1. Miss - populates cache with "Original Answer"
        res1 = cache.get_or_compute("test query", dynamic_llm)
        assert res1.response == "Original Answer"
        cache.flush_shadow()
        assert cache.stats.shadow_evaluations == 0

        # 2. Hit - dynamic_llm produces "Different Answer 2", shadow mode detects mismatch
        res2 = cache.get_or_compute("test query", dynamic_llm)
        assert res2.response == "Original Answer"  # Cached answer returned to user
        cache.flush_shadow()
        assert cache.stats.shadow_evaluations == 1
        assert cache.stats.shadow_mismatches == 1
        assert cache.stats.shadow_matches == 0
        assert cache.stats.estimated_false_hit_rate == 1.0

        # 3. Add a matching query
        cache.set("matching query", "Constant Answer")
        cache.get_or_compute("matching query", lambda p: "Constant Answer")
        cache.flush_shadow()
        assert cache.stats.shadow_evaluations == 2
        assert cache.stats.shadow_matches == 1
        assert cache.stats.shadow_mismatches == 1
        assert cache.stats.estimated_false_hit_rate == 0.5


# ---------------------------------------------------------------------------
# Live Gemini tests (2-3 live tests using real GeminiClient)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="Requires GEMINI_API_KEY")
class TestLiveGeminiIntegration:
    """Live integration tests against Google Gemini API."""

    def test_live_gemini_miss_then_hit(self) -> None:
        """Live test 1: Initial MISS calls Gemini, second call is an exact HIT."""
        try:
            client = GeminiClient()
            embedder = GeminiEmbedder()
        except Exception as exc:
            pytest.skip(f"Live Gemini initialization failed: {exc}")

        cache = SemanticCache(embedder=embedder, threshold=0.85)
        live_calls = [0]

        def call_gemini(prompt: str) -> str:
            live_calls[0] += 1
            return client.generate(prompt)

        ask = cache.cached(namespace="live-gemini:v1")(call_gemini)
        prompt = "What is the official currency of Japan? One word."

        try:
            # First call -> live MISS
            res1 = ask(prompt)
            assert live_calls[0] == 1
            assert "Yen" in res1 or "yen" in res1

            # Second call -> cache HIT
            res2 = ask(prompt)
            assert live_calls[0] == 1
            assert res1 == res2
        except GeminiAPIError as exc:
            pytest.skip(f"Live Gemini request failed / rate limited: {exc}")

    def test_live_gemini_semantic_hit(self) -> None:
        """Live test 2: Semantic paraphrase hits cache without calling Gemini."""
        try:
            client = GeminiClient()
            embedder = GeminiEmbedder()
        except Exception as exc:
            pytest.skip(f"Live Gemini initialization failed: {exc}")

        cache = SemanticCache(embedder=embedder, threshold=0.85)
        live_calls = [0]

        def call_gemini(prompt: str) -> str:
            live_calls[0] += 1
            return client.generate(prompt)

        ask = cache.cached(namespace="live-gemini:v1")(call_gemini)

        try:
            # Seed
            ask("What is the official currency of Japan? One word.")
            assert live_calls[0] == 1

            # Paraphrase
            res = ask("Official currency of Japan? One word.")
            assert live_calls[0] == 1
            assert "Yen" in res or "yen" in res
        except GeminiAPIError as exc:
            pytest.skip(f"Live Gemini request failed / rate limited: {exc}")

    def test_live_gemini_namespace_isolation(self) -> None:
        """Live test 3: Different namespace causes a MISS and generates fresh."""
        try:
            client = GeminiClient()
            embedder = GeminiEmbedder()
        except Exception as exc:
            pytest.skip(f"Live Gemini initialization failed: {exc}")

        cache = SemanticCache(embedder=embedder, threshold=0.85)
        live_calls = [0]

        def call_gemini(prompt: str) -> str:
            live_calls[0] += 1
            return client.generate(prompt)

        ask_v1 = cache.cached(namespace="live-gemini:v1")(call_gemini)
        ask_v2 = cache.cached(namespace="live-gemini:v2")(call_gemini)

        prompt = "What is the official currency of Japan? One word."
        try:
            ask_v1(prompt)
            assert live_calls[0] == 1

            ask_v2(prompt)
            assert live_calls[0] == 2
        except GeminiAPIError as exc:
            pytest.skip(f"Live Gemini request failed / rate limited: {exc}")

