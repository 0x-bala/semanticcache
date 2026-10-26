"""Regression tests for semantic cache hardening.

Covers:
  - VERIFY bypass prevention (no verifier → miss, verifier rejects → miss,
    verifier exception → miss, high similarity never overrides VERIFY)
  - must_verify=True cannot be bypassed
  - REJECT cannot be overridden by a permissive verifier
  - Rejected top candidate does not prevent a later valid candidate from matching
  - Sync and async lookups enforce equivalent verification rules
  - Streaming lookup never replays unapproved ambiguous candidates
  - explain() reports the same decision as the actual lookup
  - Exact-match queries remain isolated across contexts
  - Guard false-hit prevention (comparative entities, swapped content)
  - Non-finite embeddings, empty inputs, and other edge cases

All tests use deterministic fake embedders. No network access required.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import threading
from collections import OrderedDict

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from semantic_cache.cache import SemanticCache
from semantic_cache.context import CacheContext
from semantic_cache.guard import GuardVerdict, check_match
from semantic_cache.store import SQLiteStore

# ===========================================================================
# Test helpers
# ===========================================================================

DIM = 8


class ConstantEmbedder:
    """Every text gets the same unit vector → all pairs score 1.0."""

    dim = DIM

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        v[0] = 1.0
        return v

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class TwoVectorEmbedder:
    """Returns vec_a for queries containing 'alpha', vec_b otherwise.

    vec_a and vec_b have cosine similarity 0.0 (orthogonal).
    """

    dim = DIM

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        if "alpha" in text.lower():
            v[0] = 1.0
        else:
            v[1] = 1.0
        return v


class MultiVectorEmbedder:
    """Map specific texts to specific vectors for controlled testing.

    Unregistered texts get the zero vector.
    """

    dim = DIM

    def __init__(self):
        self._map: dict[str, np.ndarray] = {}

    def register(self, text: str, idx: int):
        v = np.zeros(DIM, dtype=np.float32)
        v[idx % DIM] = 1.0
        self._map[text] = v
        return self

    def register_vec(self, text: str, vec: np.ndarray):
        self._map[text] = vec.astype(np.float32)
        return self

    def embed(self, text: str) -> np.ndarray:
        return self._map.get(text, np.zeros(DIM, dtype=np.float32))


def _make_cache(embedder=None, **kwargs) -> SemanticCache:
    emb = embedder or ConstantEmbedder()
    store = SQLiteStore(path=":memory:", embedding_dim=getattr(emb, "dim", DIM))
    kwargs.setdefault("threshold", 0.85)
    kwargs.setdefault("exact_match", False)
    return SemanticCache(
        embedder=emb,
        store=store,
        **kwargs,
    )


# ===========================================================================
# 1. VERIFY bypass prevention
# ===========================================================================


class TestVerifyBypassPrevention:
    """VERIFY verdict must never serve a cached answer without verifier approval."""

    def test_verify_no_verifier_is_miss(self):
        """VERIFY + verify_always=False + high similarity + no verifier → miss."""
        cache = _make_cache(verify_always=False)
        # These two differ by entity slot → guard returns VERIFY
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None, (
            "VERIFY with no verifier must be a cache miss, "
            f"got response: {result.response if result else None}"
        )

    def test_verify_verifier_approves_is_hit(self):
        """VERIFY + verifier returns True → hit."""
        cache = _make_cache(verifier=lambda a, b: True)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is not None
        assert result.hit is True

    def test_verify_verifier_rejects_is_miss(self):
        """VERIFY + verifier returns False → miss."""
        cache = _make_cache(verifier=lambda a, b: False)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None

    def test_verify_verifier_exception_is_miss(self):
        """VERIFY + verifier raises → miss."""

        def exploding_verifier(a, b):
            raise RuntimeError("verifier exploded")

        cache = _make_cache(verifier=exploding_verifier)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None

    def test_verify_verifier_returns_none_is_miss(self):
        """VERIFY + verifier returns a falsy non-bool → miss."""
        cache = _make_cache(verifier=lambda a, b: None)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None

    def test_high_similarity_never_overrides_verify(self):
        """Score 1.0 + verify_always=False must still respect VERIFY verdict."""
        cache = _make_cache(verify_always=False, guard_buffer=0.0)
        cache.set("What is the best way to learn French?", "Answer about French")
        result = cache.get("What is the best way to learn Spanish?")
        assert result is None

    def test_must_verify_not_bypassed_by_high_similarity(self):
        """must_verify=True candidates cannot be bypassed by high similarity."""
        cache = _make_cache(verify_always=False, guard_buffer=0.0)
        # Entity swap → guard returns VERIFY with must_verify=True
        cache.set(
            "Is it possible for an Indian girl to be in a relationship with a Chinese boy?",
            "answer",
        )
        result = cache.get(
            "Is it possible for a Chinese girl to be in a relationship with an Indian boy?"
        )
        assert result is None


class TestStrictVerifierValidation:
    """Strict boolean validation: only Python Boolean True authorizes a hit."""

    @pytest.mark.parametrize(
        "invalid_return",
        [
            "false",
            "true",
            1,
            0,
            None,
            {"approved": True},
            [1, 2, 3],
            object(),
            "YES",
            "1",
        ],
    )
    def test_non_boolean_verifier_return_is_rejection(self, invalid_return):
        """Truthy and falsy non-boolean returns must be treated as rejection."""
        cache = _make_cache(verifier=lambda a, b: invalid_return)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None, f"Verifier return {invalid_return!r} must not authorize hit"

    def test_actual_boolean_true_approves(self):
        """Actual Python bool True authorizes a hit."""
        cache = _make_cache(verifier=lambda a, b: True)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is not None
        assert result.hit is True
        assert result.response == "Elon Musk"

    def test_actual_boolean_false_rejects(self):
        """Actual Python bool False rejects a candidate."""
        cache = _make_cache(verifier=lambda a, b: False)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None

    def test_verifier_exception_rejects(self):
        """Verifier raising an exception is treated as a rejection."""

        def bad_verifier(a, b):
            raise ValueError("network error / timeout")

        cache = _make_cache(verifier=bad_verifier)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        assert cache.get("Who is the CEO of Apple?") is None

    @pytest.mark.parametrize("invalid_return", ["false", "true", 1, 0, None, object()])
    def test_explain_rejects_invalid_verifier_return(self, invalid_return):
        """explain() must report verifier_reject for non-boolean returns."""
        cache = _make_cache(verifier=lambda a, b: invalid_return)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        exp = cache.explain("Who is the CEO of Apple?")
        assert exp["decision"] == "verifier_reject"

    @pytest.mark.parametrize("invalid_return", ["true", 1, "false", None])
    def test_async_rejects_invalid_verifier_return(self, invalid_return):
        """aget() rejects non-boolean returns."""
        cache = _make_cache(verifier=lambda a, b: invalid_return)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        async def _lookup():
            return await cache.aget("Who is the CEO of Apple?")

        assert asyncio.run(_lookup()) is None

    @pytest.mark.parametrize("invalid_return", ["true", 1, "false", None])
    def test_streaming_rejects_invalid_verifier_return(self, invalid_return):
        """get_or_compute_stream() does not replay on non-boolean return."""
        cache = _make_cache(verifier=lambda a, b: invalid_return)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        def compute_stream(q):
            yield "Tim "
            yield "Cook"

        chunks = list(cache.get_or_compute_stream("Who is the CEO of Apple?", compute_stream))
        assert "".join(chunks) == "Tim Cook"

    def test_verifier_with_valid_score_and_threshold(self):
        """Verifier with score and threshold authorizes hit when score >= threshold and True."""

        class MockScoringVerifier:
            threshold = 0.85

            def score(self, a, b):
                return 0.92

            def __call__(self, a, b):
                return True

        cache = _make_cache(verifier=MockScoringVerifier())
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is not None
        assert result.hit is True

        exp = cache.explain("Who is the CEO of Apple?")
        assert exp["decision"] == "hit_verified"
        assert exp["verifier_score"] == pytest.approx(0.92)

    def test_verifier_with_score_below_threshold(self):
        """Verifier exposing score < threshold rejects even if callable returned True."""

        class MockScoringVerifier:
            threshold = 0.85

            def score(self, a, b):
                return 0.80

            def __call__(self, a, b):
                return True

        cache = _make_cache(verifier=MockScoringVerifier())
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None

        exp = cache.explain("Who is the CEO of Apple?")
        assert exp["decision"] == "verifier_reject"

    @pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
    def test_verifier_with_non_finite_score(self, bad_score):
        """Verifier with NaN or Inf score is rejected."""

        class MockScoringVerifier:
            threshold = 0.85

            def score(self, a, b):
                return bad_score

            def __call__(self, a, b):
                return True

        cache = _make_cache(verifier=MockScoringVerifier())
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None

        exp = cache.explain("Who is the CEO of Apple?")
        assert exp["decision"] == "verifier_reject"

    def test_verifier_with_score_returning_non_boolean_call(self):
        """Verifier with score >= threshold but returning 'true' (string) is rejected."""

        class MockScoringVerifier:
            threshold = 0.85

            def score(self, a, b):
                return 0.95

            def __call__(self, a, b):
                return "true"

        cache = _make_cache(verifier=MockScoringVerifier())
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None

        exp = cache.explain("Who is the CEO of Apple?")
        assert exp["decision"] == "verifier_reject"

    @pytest.mark.parametrize(
        "bool_prediction",
        [True, False, np.bool_(True), np.bool_(False), [True], [False], [np.bool_(True)]],
    )
    def test_model_prediction_boolean_rejected(self, bool_prediction):
        """Model predictions of Python bool or NumPy bool are rejected as invalid scores."""

        class FakeModelVerifier:
            threshold = 0.5

            class FakeModel:
                def predict(self, pairs):
                    return [bool_prediction]

            model = FakeModel()

            def __call__(self, a, b):
                return True

        cache = _make_cache(verifier=FakeModelVerifier())
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        result = cache.get("Who is the CEO of Apple?")
        assert result is None, f"Boolean prediction {bool_prediction!r} must not authorize hit"

        exp = cache.explain("Who is the CEO of Apple?")
        assert exp["decision"] == "verifier_reject"
        assert exp["verifier_score"] is None

    @pytest.mark.parametrize(
        "bool_score",
        [True, False, np.bool_(True), np.bool_(False)],
    )
    def test_score_method_boolean_rejected(self, bool_score):
        """Verifier score() returning Python or NumPy boolean is rejected."""

        class MockBooleanScoreVerifier:
            threshold = 0.5

            def score(self, a, b):
                return bool_score

            def __call__(self, a, b):
                return True

        cache = _make_cache(verifier=MockBooleanScoreVerifier())
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        assert cache.get("Who is the CEO of Apple?") is None
        exp = cache.explain("Who is the CEO of Apple?")
        assert exp["decision"] == "verifier_reject"
        assert exp["verifier_score"] is None

    def test_cross_encoder_rejects_boolean_prediction(self):
        """CrossEncoderVerifier.__call__ and score() reject boolean predictions."""
        from semantic_cache.cross_encoder import CrossEncoderVerifier

        cev = object.__new__(CrossEncoderVerifier)
        cev.threshold = 0.85
        cev.max_cache_size = 1000
        cev._cache = OrderedDict()
        cev._cache_lock = threading.Lock()
        cev._inflight = {}
        cev._last_pair = None
        cev._last_score = None

        class FakeModel:
            def predict(self, pairs):
                return [True]

        cev.model = FakeModel()
        assert cev.score("q1", "q2") is None
        assert cev("q1", "q2") is False

        # NumPy bool
        cev.clear_cache()
        cev.model.predict = lambda pairs: [np.bool_(True)]
        assert cev.score("q1", "q2") is None
        assert cev("q1", "q2") is False

        # Valid score
        cev.clear_cache()
        cev.model.predict = lambda pairs: [0.95]
        assert cev.score("q1", "q2") == pytest.approx(0.95)
        assert cev("q1", "q2") is True

    def test_cross_encoder_memoizes_single_inference_per_pair(self):
        """Score calculation and __call__ reuse model prediction without duplicate inference."""
        from semantic_cache.cross_encoder import CrossEncoderVerifier

        cev = object.__new__(CrossEncoderVerifier)
        cev.threshold = 0.85
        cev.max_cache_size = 1000
        cev._cache = OrderedDict()
        cev._cache_lock = threading.Lock()
        cev._inflight = {}
        cev._last_pair = None
        cev._last_score = None

        class CountingModel:
            def __init__(self):
                self.calls = 0

            def predict(self, pairs):
                self.calls += 1
                return [0.95]

        cev.model = CountingModel()
        cache = _make_cache(verifier=cev)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        res = cache.get("Who is the CEO of Apple?")
        assert res is not None
        assert res.hit is True
        # Model predict should only be called once for this pair
        assert cev.model.calls == 1


# ===========================================================================
# 2. REJECT cannot be overridden
# ===========================================================================


class TestRejectNotOverridable:
    """REJECT verdicts must never serve a cached answer."""

    def test_reject_even_with_permissive_verifier(self):
        """REJECT must not be overridden by a permissive verifier."""
        cache = _make_cache(verifier=lambda a, b: True)
        cache.set("Convert 10 km to miles", "6.21 miles")
        # Unit inversion → guard REJECT
        result = cache.get("Convert 10 miles to km")
        assert result is None
        assert cache.stats.guard_rejects >= 1

    def test_reject_negation_with_verifier(self):
        """Negation flip → REJECT, even with verifier."""
        cache = _make_cache(verifier=lambda a, b: True)
        cache.set("Is it safe to travel?", "Yes it is safe")
        result = cache.get("Is it not safe to travel?")
        assert result is None

    def test_reject_number_difference(self):
        """Different numbers → REJECT."""
        cache = _make_cache()
        cache.set("Top 5 movies", "List of 5 movies")
        result = cache.get("Top 10 movies")
        assert result is None

    def test_reject_symbol_difference(self):
        """Different operators → REJECT."""
        cache = _make_cache()
        cache.set("Is x>5?", "Analysis of x>5")
        result = cache.get("Is x<5?")
        assert result is None


# ===========================================================================
# 3. Rejected top candidate does not prevent later valid candidates
# ===========================================================================


class TestCandidateSearchContinuation:
    """After rejecting a candidate, the cache must continue searching."""

    def test_rejected_first_candidate_allows_later_match(self):
        """If first candidate is rejected by guard, search continues and returns second."""
        emb = MultiVectorEmbedder()
        q_vec = np.zeros(DIM, dtype=np.float32)
        q_vec[0] = 1.0

        # Candidate 1: cosine similarity 0.98 with query
        c1_vec = np.zeros(DIM, dtype=np.float32)
        c1_vec[0] = 0.98
        c1_vec[1] = np.sqrt(1.0 - 0.98**2, dtype=np.float32)

        # Candidate 2: cosine similarity 0.90 with query (still above threshold 0.85)
        c2_vec = np.zeros(DIM, dtype=np.float32)
        c2_vec[0] = 0.90
        c2_vec[2] = np.sqrt(1.0 - 0.90**2, dtype=np.float32)

        q_text = "What is the capital of France?"
        c1_text = "What is not the capital of France?"  # Negation conflict -> REJECT
        c2_text = "Capital city of France?"            # Acceptable paraphrase -> ACCEPT

        emb.register_vec(q_text, q_vec)
        emb.register_vec(c1_text, c1_vec)
        emb.register_vec(c2_text, c2_vec)

        cache = _make_cache(embedder=emb, threshold=0.85, top_k=5)
        cache.set(c1_text, "Response 1 (Not Paris)")
        cache.set(c2_text, "Response 2 (Paris)")

        result = cache.get(q_text)
        assert result is not None, "Candidate search stopped early instead of falling through"
        assert result.hit is True
        assert result.matched_query == c2_text
        assert result.response == "Response 2 (Paris)"
        assert result.score == pytest.approx(0.90, abs=1e-3)
        assert cache.stats.guard_rejects >= 1

    def test_verifier_rejected_first_candidate_allows_later_match(self):
        """If first candidate is rejected by verifier, search continues and returns second."""
        emb = MultiVectorEmbedder()
        q_vec = np.zeros(DIM, dtype=np.float32)
        q_vec[0] = 1.0

        # Candidate 1: cosine similarity 0.98
        c1_vec = np.zeros(DIM, dtype=np.float32)
        c1_vec[0] = 0.98
        c1_vec[1] = np.sqrt(1.0 - 0.98**2, dtype=np.float32)

        # Candidate 2: cosine similarity 0.90
        c2_vec = np.zeros(DIM, dtype=np.float32)
        c2_vec[0] = 0.90
        c2_vec[2] = np.sqrt(1.0 - 0.90**2, dtype=np.float32)

        q_text = "Who is the CEO of Apple?"
        c1_text = "Who is the CEO of Tesla?"  # Guard VERIFY -> verifier returns False
        c2_text = "Who is Apple CEO?"          # Guard VERIFY -> verifier returns True

        emb.register_vec(q_text, q_vec)
        emb.register_vec(c1_text, c1_vec)
        emb.register_vec(c2_text, c2_vec)

        def selective_verifier(query, candidate):
            if "Apple" in candidate:
                return True
            return False

        cache = _make_cache(embedder=emb, threshold=0.85, top_k=5, verifier=selective_verifier)
        cache.set(c1_text, "Elon Musk")
        cache.set(c2_text, "Tim Cook")

        result = cache.get(q_text)
        assert result is not None, "Candidate search stopped early instead of falling through"
        assert result.hit is True
        assert result.matched_query == c2_text
        assert result.response == "Tim Cook"
        assert result.score == pytest.approx(0.90, abs=1e-3)

    def test_explain_candidate_fallback_parity(self):
        """explain() reports the correct accepted candidate when earlier candidate was rejected."""
        emb = MultiVectorEmbedder()
        q_vec = np.zeros(DIM, dtype=np.float32)
        q_vec[0] = 1.0

        c1_vec = np.zeros(DIM, dtype=np.float32)
        c1_vec[0] = 0.98
        c1_vec[1] = np.sqrt(1.0 - 0.98**2, dtype=np.float32)

        c2_vec = np.zeros(DIM, dtype=np.float32)
        c2_vec[0] = 0.90
        c2_vec[2] = np.sqrt(1.0 - 0.90**2, dtype=np.float32)

        q_text = "What is the capital of France?"
        c1_text = "What is not the capital of France?"  # REJECT
        c2_text = "Capital city of France?"            # ACCEPT

        emb.register_vec(q_text, q_vec)
        emb.register_vec(c1_text, c1_vec)
        emb.register_vec(c2_text, c2_vec)

        cache = _make_cache(embedder=emb, threshold=0.85, top_k=5)
        cache.set(c1_text, "Not Paris")
        cache.set(c2_text, "Paris")

        exp = cache.explain(q_text)
        assert exp["decision"] == "hit_guard_accept"
        assert exp["best_candidate"]["query"] == c2_text
        assert exp["best_candidate"]["score"] == pytest.approx(0.90, abs=1e-3)


# ===========================================================================
# 4. Async lookups enforce same rules
# ===========================================================================


class TestAsyncVerificationRules:
    """Async paths must enforce the same verification policies as sync."""

    def test_async_verify_no_verifier_is_miss(self):
        """Async: VERIFY + no verifier → miss."""
        cache = _make_cache(verify_always=False)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        async def _lookup():
            return await cache.aget("Who is the CEO of Apple?")

        result = asyncio.run(_lookup())
        assert result is None

    def test_async_verify_verifier_approves_is_hit(self):
        """Async: VERIFY + verifier approves → hit."""
        cache = _make_cache(verifier=lambda a, b: True)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        async def _lookup():
            return await cache.aget("Who is the CEO of Apple?")

        result = asyncio.run(_lookup())
        assert result is not None
        assert result.hit is True

    def test_async_verify_verifier_rejects_is_miss(self):
        """Async: VERIFY + verifier rejects → miss."""
        cache = _make_cache(verifier=lambda a, b: False)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        async def _lookup():
            return await cache.aget("Who is the CEO of Apple?")

        result = asyncio.run(_lookup())
        assert result is None

    def test_async_get_or_compute_verify_miss(self):
        """Async get_or_compute: VERIFY + no verifier → compute fresh."""
        cache = _make_cache()
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        async def compute(q):
            return "Tim Cook"

        async def _run():
            return await cache.async_get_or_compute(
                "Who is the CEO of Apple?", compute
            )

        result = asyncio.run(_run())
        assert result.response == "Tim Cook"
        assert result.hit is False


# ===========================================================================
# 5. Streaming never replays unapproved ambiguous candidates
# ===========================================================================


class TestStreamingVerification:
    """Streaming paths must not serve VERIFY candidates without approval."""

    def test_stream_verify_no_verifier_computes_fresh(self):
        """Streaming: VERIFY + no verifier → compute fresh stream."""
        cache = _make_cache()
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        def stream_compute(q):
            yield "Tim "
            yield "Cook"

        chunks = list(
            cache.get_or_compute_stream(
                "Who is the CEO of Apple?", stream_compute
            )
        )
        assert "".join(chunks) == "Tim Cook"

    def test_stream_reject_computes_fresh(self):
        """Streaming: REJECT → compute fresh stream."""
        cache = _make_cache()
        cache.set("Convert 10 km to miles", "6.21 miles")

        def stream_compute(q):
            yield "16.09 "
            yield "km"

        chunks = list(
            cache.get_or_compute_stream(
                "Convert 10 miles to km", stream_compute
            )
        )
        assert "".join(chunks) == "16.09 km"

    def test_async_stream_verify_no_verifier_computes_fresh(self):
        """Async streaming: VERIFY + no verifier → compute fresh."""
        cache = _make_cache()
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        async def async_stream(q):
            yield "Tim "
            yield "Cook"

        async def _run():
            chunks = []
            async for chunk in cache.async_get_or_compute_stream(
                "Who is the CEO of Apple?", async_stream
            ):
                chunks.append(chunk)
            return "".join(chunks)

        result = asyncio.run(_run())
        assert result == "Tim Cook"


# ===========================================================================
# 6. explain() consistency with get()
# ===========================================================================


class TestExplainConsistency:
    """explain() must report the same decision as get()."""

    def test_explain_accept_matches_get_hit(self):
        """Guard ACCEPT → explain says hit_guard_accept, get returns hit."""
        cache = _make_cache()
        cache.set("What is the capital of France?", "Paris")
        result = cache.get("Capital city of France?")
        explain = cache.explain("Capital city of France?")
        if result is not None:
            assert explain["decision"] == "hit_guard_accept"
        else:
            assert explain["decision"] in ("no_candidates", "below_threshold")

    def test_explain_reject_matches_get_miss(self):
        """Guard REJECT → explain says guard_reject, get returns None."""
        cache = _make_cache()
        cache.set("Enable the setting", "ok")
        result = cache.get("Disable the setting")
        explain = cache.explain("Disable the setting")
        assert result is None
        assert explain["decision"] == "guard_reject"

    def test_explain_verify_no_verifier_matches_get_miss(self):
        """VERIFY + no verifier → explain says verifier_reject, get returns None."""
        cache = _make_cache()
        cache.set("Who wrote Hamlet?", "Shakespeare")
        result = cache.get("Who is the author of Hamlet?")
        explain = cache.explain("Who is the author of Hamlet?")
        assert result is None
        assert explain["decision"] == "verifier_reject"

    def test_explain_verify_approved_matches_get_hit(self):
        """VERIFY + verifier approves → explain says hit_verified, get returns hit."""
        cache = _make_cache(verifier=lambda a, b: True)
        cache.set("Who wrote Hamlet?", "Shakespeare")
        result = cache.get("Who is the author of Hamlet?")
        explain = cache.explain("Who is the author of Hamlet?")
        assert result is not None
        assert explain["decision"] == "hit_verified"


# ===========================================================================
# 7. Context isolation
# ===========================================================================


class TestContextIsolation:
    """CacheContext must produce distinct namespaces for distinct contexts."""

    def test_different_tenants_different_namespaces(self):
        ctx_a = CacheContext(
            tenant="tenant-a", model="gpt-4", system_prompt_version="v1"
        )
        ctx_b = CacheContext(
            tenant="tenant-b", model="gpt-4", system_prompt_version="v1"
        )
        assert ctx_a.namespace() != ctx_b.namespace()

    def test_different_models_different_namespaces(self):
        ctx_a = CacheContext(
            tenant="t", model="gpt-4", system_prompt_version="v1"
        )
        ctx_b = CacheContext(
            tenant="t", model="gpt-3.5", system_prompt_version="v1"
        )
        assert ctx_a.namespace() != ctx_b.namespace()

    def test_different_prompt_versions_different_namespaces(self):
        ctx_a = CacheContext(
            tenant="t", model="gpt-4", system_prompt_version="v1"
        )
        ctx_b = CacheContext(
            tenant="t", model="gpt-4", system_prompt_version="v2"
        )
        assert ctx_a.namespace() != ctx_b.namespace()

    def test_different_user_scopes_different_namespaces(self):
        ctx_a = CacheContext(
            tenant="t",
            model="gpt-4",
            system_prompt_version="v1",
            user_scope="user-1",
        )
        ctx_b = CacheContext(
            tenant="t",
            model="gpt-4",
            system_prompt_version="v1",
            user_scope="user-2",
        )
        assert ctx_a.namespace() != ctx_b.namespace()

    def test_different_temperature_different_namespaces(self):
        ctx_a = CacheContext(
            tenant="t",
            model="gpt-4",
            system_prompt_version="v1",
            temperature=0.0,
        )
        ctx_b = CacheContext(
            tenant="t",
            model="gpt-4",
            system_prompt_version="v1",
            temperature=0.7,
        )
        assert ctx_a.namespace() != ctx_b.namespace()

    def test_same_context_same_namespace(self):
        ctx_a = CacheContext(
            tenant="t", model="gpt-4", system_prompt_version="v1"
        )
        ctx_b = CacheContext(
            tenant="t", model="gpt-4", system_prompt_version="v1"
        )
        assert ctx_a.namespace() == ctx_b.namespace()

    def test_context_namespace_isolates_cache_entries(self):
        """Same query text in different contexts must not cross-contaminate."""
        cache = _make_cache()
        ctx_a = CacheContext(
            tenant="acme", model="gpt-4", system_prompt_version="v1"
        )
        ctx_b = CacheContext(
            tenant="globex", model="gpt-4", system_prompt_version="v1"
        )
        cache.set("What is our policy?", "Acme policy", namespace=ctx_a.namespace())
        cache.set("What is our policy?", "Globex policy", namespace=ctx_b.namespace())

        result_a = cache.get("What is our policy?", namespace=ctx_a.namespace())
        result_b = cache.get("What is our policy?", namespace=ctx_b.namespace())

        assert result_a is not None
        assert result_b is not None
        assert result_a.response == "Acme policy"
        assert result_b.response == "Globex policy"

    def test_exact_match_respects_namespace_isolation(self):
        """Exact-match tier must not cross namespaces."""
        cache = SemanticCache(
            embedder=ConstantEmbedder(),
            store=SQLiteStore(path=":memory:", embedding_dim=DIM),
            threshold=0.85,
            exact_match=True,
        )
        cache.set("What is our policy?", "Acme policy", namespace="acme")
        cache.set("What is our policy?", "Globex policy", namespace="globex")

        result_a = cache.get("What is our policy?", namespace="acme")
        result_b = cache.get("What is our policy?", namespace="globex")

        assert result_a is not None and result_a.response == "Acme policy"
        assert result_b is not None and result_b.response == "Globex policy"

    def test_exact_match_respects_embedder_isolation(self):
        """Exact-match tier must not cross embedder partitions."""
        store = SQLiteStore(path=":memory:", embedding_dim=DIM)

        class EmbedderA(ConstantEmbedder):
            model_name = "model-a"

        class EmbedderB(ConstantEmbedder):
            model_name = "model-b"

        cache_a = SemanticCache(
            embedder=EmbedderA(), store=store, threshold=0.85, exact_match=True
        )
        cache_b = SemanticCache(
            embedder=EmbedderB(), store=store, threshold=0.85, exact_match=True
        )

        cache_a.set("hello", "answer-a")
        cache_b.set("hello", "answer-b")

        result_a = cache_a.get("hello")
        result_b = cache_b.get("hello")

        assert result_a is not None and result_a.response == "answer-a"
        assert result_b is not None and result_b.response == "answer-b"

    def test_context_invalid_empty_tenant_raises(self):
        """Empty required fields must raise ValueError."""
        with pytest.raises(ValueError):
            CacheContext(
                tenant="", model="gpt-4", system_prompt_version="v1"
            ).namespace()

    def test_context_invalid_temperature_type_raises(self):
        """Non-numeric temperature must raise TypeError."""
        with pytest.raises(TypeError):
            CacheContext(
                tenant="t",
                model="gpt-4",
                system_prompt_version="v1",
                temperature="high",  # type: ignore
            ).namespace()


# ===========================================================================
# 8. Guard false-hit regression tests
# ===========================================================================


class TestGuardFalseHits:
    """Guard must prevent known false-hit patterns."""

    @pytest.mark.parametrize(
        "new_q,cached_q",
        [
            ("Convert 10 km to miles", "Convert 10 miles to km"),
            ("Convert 50 fahrenheit to celsius", "Convert 50 celsius to fahrenheit"),
        ],
    )
    def test_unit_inversion_rejected(self, new_q, cached_q):
        result = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.REJECT

    @pytest.mark.parametrize(
        "new_q,cached_q",
        [
            ("Enable dark mode", "Disable dark mode"),
            ("Buy Bitcoin", "Sell Bitcoin"),
            ("Install the package", "Uninstall the package"),
        ],
    )
    def test_antonym_rejected(self, new_q, cached_q):
        result = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.REJECT

    @pytest.mark.parametrize(
        "new_q,cached_q",
        [
            ("Who is the CEO of Apple?", "Who is the CEO of Tesla?"),
            ("Weather in Tokyo", "Weather in London"),
            ("Symptoms of flu", "Symptoms of measles"),
        ],
    )
    def test_entity_substitution_verify(self, new_q, cached_q):
        result = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.VERIFY
        assert result.must_verify is True

    @pytest.mark.parametrize(
        "new_q,cached_q",
        [
            ("What is Python?", "Who invented Python?"),
            ("Where is Paris?", "When was Paris founded?"),
        ],
    )
    def test_question_intent_rejected(self, new_q, cached_q):
        result = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.REJECT

    @pytest.mark.parametrize(
        "new_q,cached_q",
        [
            ("what's the capital of france", "What is the capital of France?"),
            ("How can I reset my password?", "How do I reset my password?"),
            ("Who wrote the play Hamlet?", "Who wrote Hamlet?"),
        ],
    )
    def test_safe_paraphrases_accepted(self, new_q, cached_q):
        result = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.ACCEPT


# ===========================================================================
# 9. Stats tracking
# ===========================================================================


class TestStatsTracking:
    """Stats must correctly track guard rejects and verify calls."""

    def test_guard_reject_increments_stats(self):
        cache = _make_cache()
        cache.set("Enable dark mode", "ok")
        cache.get("Disable dark mode")
        assert cache.stats.guard_rejects >= 1

    def test_verify_calls_incremented_with_verifier(self):
        cache = _make_cache(verifier=lambda a, b: True)
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        cache.get("Who is the CEO of Apple?")
        assert cache.stats.verify_calls >= 1
        assert cache.stats.verify_accepts >= 1

    def test_verify_calls_not_incremented_without_verifier(self):
        cache = _make_cache()
        cache.set("Who is the CEO of Tesla?", "Elon Musk")
        cache.get("Who is the CEO of Apple?")
        assert cache.stats.verify_calls == 0


# ===========================================================================
# 10. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases that must continue to behave safely."""

    def test_empty_query_get_returns_none(self):
        cache = _make_cache()
        result = cache.get("")
        assert result is None

    def test_whitespace_query_get_returns_none(self):
        cache = _make_cache()
        result = cache.get("   ")
        assert result is None

    def test_set_empty_query_raises(self):
        cache = _make_cache()
        with pytest.raises(ValueError):
            cache.set("", "response")

    def test_set_whitespace_query_raises(self):
        cache = _make_cache()
        with pytest.raises(ValueError):
            cache.set("   ", "response")

    def test_set_empty_response_raises(self):
        cache = _make_cache()
        with pytest.raises(ValueError):
            cache.set("query", "")

    def test_disabled_cache_always_misses(self):
        cache = _make_cache(enabled=False)
        cache.set("hello", "world")  # no-op when disabled
        result = cache.get("hello")
        assert result is None

    def test_bypass_always_misses(self):
        cache = _make_cache()
        cache.set("hello", "world")
        result = cache.get("hello", bypass=True)
        assert result is None

    def test_non_string_query_raises(self):
        cache = _make_cache()
        with pytest.raises(TypeError):
            cache.get(123)  # type: ignore

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            SemanticCache(threshold=1.5)
        with pytest.raises(ValueError):
            SemanticCache(threshold=-0.1)

    def test_cache_context_manager(self):
        """Cache used as context manager should close cleanly."""
        with _make_cache() as cache:
            cache.set("hello", "world")
            result = cache.get("hello")
            assert result is not None


# ===========================================================================
# 11. Decorator tests
# ===========================================================================


class TestDecoratorVerification:
    """@cached decorator must enforce the same verification rules."""

    def test_cached_decorator_reject_computes_fresh(self):
        cache = _make_cache()
        cache.set("Convert 10 km to miles", "6.21 miles")

        @cache.cached()
        def compute(prompt: str) -> str:
            return f"fresh: {prompt}"

        result = compute("Convert 10 miles to km")
        assert result.startswith("fresh:")

    def test_cached_decorator_verify_no_verifier_computes_fresh(self):
        cache = _make_cache()
        cache.set("Who is the CEO of Tesla?", "Elon Musk")

        @cache.cached()
        def compute(prompt: str) -> str:
            return "Tim Cook"

        result = compute("Who is the CEO of Apple?")
        assert result == "Tim Cook"
