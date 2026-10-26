"""Tests for allow_question_intent_mismatch on SemanticCache and check_match."""

from __future__ import annotations

import importlib.util

import pytest

from semantic_cache import HashingEmbedder, SemanticCache, SQLiteStore
from semantic_cache.guard import GuardVerdict, check_match

_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None

# A paraphrase pair that the "Question intent differs" rule would otherwise REJECT.
_INTENT_PAIR = ("What is the capital of France?", "Which city is the capital of France?")

# Hard negatives that must remain misses even when the question-intent rule is
# downgraded to VERIFY and a real verifier is configured.
HARD_NEGATIVES = [
    ("Who founded Apple?", "When was Apple founded?"),
    ("What is the capital of France?", "What is the population of France?"),
    ("How tall is Mount Everest?", "How old is Mount Everest?"),
    ("Who wrote Hamlet?", "When was Hamlet written?"),
    ("Who wrote Hamlet?", "Why did Shakespeare write Hamlet?"),
    (
        "What is the boiling point of water in Celsius?",
        "What is the freezing point of water in Celsius?",
    ),
]


# ---------------------------------------------------------------------------
# check_match: flag off keeps REJECT; flag on returns VERIFY
# ---------------------------------------------------------------------------


class TestCheckMatchQuestionIntent:
    def test_flag_off_rejects(self) -> None:
        result = check_match(_INTENT_PAIR[1], _INTENT_PAIR[0])
        assert result.verdict == GuardVerdict.REJECT
        assert result.reason.startswith("Question intent differs")

    def test_flag_on_returns_verify(self) -> None:
        result = check_match(
            _INTENT_PAIR[1], _INTENT_PAIR[0], allow_question_intent_mismatch=True
        )
        assert result.verdict == GuardVerdict.VERIFY
        assert result.must_verify is True
        assert result.reason.startswith("Question intent differs")

    def test_flag_on_does_not_downgrade_other_rejects(self) -> None:
        # Negation flip must stay REJECT regardless of the flag.
        result = check_match(
            "Enable the setting",
            "Disable the setting",
            allow_question_intent_mismatch=True,
        )
        assert result.verdict == GuardVerdict.REJECT


# ---------------------------------------------------------------------------
# SemanticCache: flag off REJECT, flag on without verifier still REJECT,
# flag on with verifier adjudicates.
# ---------------------------------------------------------------------------


class TestSemanticCacheQuestionIntent:
    def _cache(self, **kwargs) -> SemanticCache:
        return SemanticCache(
            embedder=HashingEmbedder(),
            store=SQLiteStore(":memory:"),
            threshold=0.85,
            exact_match=False,
            **kwargs,
        )

    def test_flag_off_rejects(self) -> None:
        cache = self._cache(allow_question_intent_mismatch=False)
        cache.set(_INTENT_PAIR[0], "answer")
        assert cache.get(_INTENT_PAIR[1]) is None

    def test_flag_on_without_verifier_still_rejects(self) -> None:
        # No verifier: an unverified VERIFY is still a miss.
        cache = self._cache(allow_question_intent_mismatch=True)
        cache.set(_INTENT_PAIR[0], "answer")
        assert cache.get(_INTENT_PAIR[1]) is None

    def test_flag_on_with_verifier_accepts_high_score(self) -> None:
        cache = self._cache(
            allow_question_intent_mismatch=True,
            verifier=lambda _a, _b: True,
        )
        cache.set(_INTENT_PAIR[0], "answer")
        result = cache.get(_INTENT_PAIR[1])
        assert result is not None and result.hit is True

    def test_flag_on_with_verifier_rejects_low_score(self) -> None:
        cache = self._cache(
            allow_question_intent_mismatch=True,
            verifier=lambda _a, _b: False,
        )
        cache.set(_INTENT_PAIR[0], "answer")
        assert cache.get(_INTENT_PAIR[1]) is None

    def test_default_flag_is_false(self) -> None:
        cache = self._cache()
        assert cache.allow_question_intent_mismatch is False


# ---------------------------------------------------------------------------
# Hard-negative pairs stay misses with the real CrossEncoderVerifier.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SENTENCE_TRANSFORMERS, reason="needs sentence-transformers")
class TestHardNegativesWithRealVerifier:
    """The hard negatives from the intent table must remain misses even when
    the question-intent rule is downgraded to VERIFY."""

    @pytest.mark.parametrize(
        "cached_q,new_q",
        HARD_NEGATIVES,
        ids=[n for _, n in HARD_NEGATIVES],
    )
    def test_hard_negative_is_miss(self, real_verifier_cache, cached_q, new_q) -> None:
        assert real_verifier_cache.get(new_q) is None


@pytest.fixture(scope="module")
def real_verifier_cache():
    """A cache with the real CrossEncoderVerifier and the intent rule
    downgraded, seeded with every hard-negative pair. Module-scoped so the
    model is loaded once per session (avoids the class-scoped fixture
    deprecation warning)."""
    from semantic_cache.cross_encoder import CrossEncoderVerifier

    verifier = CrossEncoderVerifier(threshold=0.9)
    cache = SemanticCache(
        embedder=HashingEmbedder(),
        store=SQLiteStore(":memory:"),
        threshold=0.85,
        exact_match=False,
        verifier=verifier,
        allow_question_intent_mismatch=True,
    )
    for cached_q, _ in HARD_NEGATIVES:
        cache.set(cached_q, f"answer to {cached_q}")
    yield cache
    cache.close(close_store=True)