"""Unit tests for SemanticCache hit guard (semantic_cache/guard.py).

Tests all 5 rules (a-e) across at least 30 diverse query pairs.
"""
from __future__ import annotations

import pytest

from semantic_cache.guard import (
    GuardResult,
    GuardVerdict,
    check_match,
    normalize_text,
)

# ---------------------------------------------------------------------------
# Test Data: 36 query pairs
# ---------------------------------------------------------------------------

ACCEPT_PAIRS = [
    # Required core pairs
    ("what's the capital of france", "What is the capital of France?"),
    ("How can I reset my password?", "How do I reset my password?"),
    ("Who wrote the play Hamlet?", "Who wrote Hamlet?"),
    ("Capital city of France?", "What is the capital of France?"),
    # Additional pairs
    ("Please tell me the capital of France", "What is the capital of France?"),
    ("Explain how to reset my password kindly", "How to reset my password"),
    ("Who is the CEO of Tesla please?", "Who is the CEO of Tesla?"),
    ("Convert 10 km to miles please", "Convert 10 km to miles"),
    ("Where's my order?", "Where is my order?"),
    ("Define machine learning", "What is the definition of machine learning?"),
]

VERIFY_PAIRS = [
    # Required core pairs
    ("What is the capital of Spain?", "What is the capital of France?"),
    ("How do I reset my router?", "How do I reset my password?"),
    ("Who wrote Macbeth?", "Who wrote Hamlet?"),
    ("Who is the CEO of Apple?", "Who is the CEO of Tesla?"),
    # Additional slot-substitution pairs
    ("What is the population of Canada?", "What is the population of Germany?"),
    ("Weather in Tokyo", "Weather in London"),
    ("Symptoms of flu", "Symptoms of measles"),
    ("How to cook salmon", "How to cook chicken"),
]

REJECT_NUMBERS_PAIRS = [
    ("Top 5 movies", "Top 10 movies"),
    ("Convert 10 km to miles", "Convert 20 km to miles"),
    ("I have two cats", "I have three cats"),
    ("Chapter one summary", "Chapter five summary"),
    ("Flight 101 status", "Flight 202 status"),
    ("Price of 10 apples", "Price of 5 apples"),
]

REJECT_UNIT_INVERSION_PAIRS = [
    # Required core pair
    ("Convert 10 miles to km", "Convert 10 km to miles"),
    # Additional unit sequence pairs
    ("Convert 50 fahrenheit to celsius", "Convert 50 celsius to fahrenheit"),
    ("Exchange 100 usd to eur", "Exchange 100 eur to usd"),
    ("Change 5 kg to pounds", "Change 5 pounds to kg"),
]

REJECT_NEGATION_SCOPE_PAIRS = [
    # Required core pair
    ("refundable ticket policy", "non-refundable ticket policy"),
    # Additional negation & scope pairs
    ("Is it safe to travel?", "Is it not safe to travel?"),
    ("Can I enter without a visa?", "Can I enter with a visa?"),
    ("Authorized personnel only", "Unauthorized personnel"),
    ("Expected delivery date", "Unexpected delivery date"),
    ("Never restart the server", "Restart the server"),
    ("Limited warranty", "Unlimited warranty"),
    ("Only registered users can access", "Registered users can access"),
]


class TestGuardRules:
    """Comprehensive unit test suite for check_match."""

    @pytest.mark.parametrize(("new_q", "cached_q"), ACCEPT_PAIRS)
    def test_rule_e_accept_paraphrases(self, new_q: str, cached_q: str) -> None:
        result: GuardResult = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.ACCEPT, (
            f"Expected ACCEPT for {new_q!r} ~ {cached_q!r}, got {result.verdict} ({result.reason})"
        )

    @pytest.mark.parametrize(("new_q", "cached_q"), VERIFY_PAIRS)
    def test_rule_d_verify_different_slot_values(self, new_q: str, cached_q: str) -> None:
        result: GuardResult = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.VERIFY, (
            f"Expected VERIFY for {new_q!r} ~ {cached_q!r}, got {result.verdict} ({result.reason})"
        )

    @pytest.mark.parametrize(("new_q", "cached_q"), REJECT_NUMBERS_PAIRS)
    def test_rule_a_reject_number_differences(self, new_q: str, cached_q: str) -> None:
        result: GuardResult = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.REJECT, (
            f"Expected REJECT for {new_q!r} ~ {cached_q!r}, got {result.verdict} ({result.reason})"
        )
        assert "number" in result.reason.lower()

    @pytest.mark.parametrize(("new_q", "cached_q"), REJECT_UNIT_INVERSION_PAIRS)
    def test_rule_b_reject_unit_order_inversion(self, new_q: str, cached_q: str) -> None:
        result: GuardResult = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.REJECT, (
            f"Expected REJECT for {new_q!r} ~ {cached_q!r}, got {result.verdict} ({result.reason})"
        )
        assert "unit" in result.reason.lower() or "inverted" in result.reason.lower()

    @pytest.mark.parametrize(("new_q", "cached_q"), REJECT_NEGATION_SCOPE_PAIRS)
    def test_rule_c_reject_negation_and_prefix_flips(self, new_q: str, cached_q: str) -> None:
        result: GuardResult = check_match(new_q, cached_q)
        assert result.verdict == GuardVerdict.REJECT, (
            f"Expected REJECT for {new_q!r} ~ {cached_q!r}, got {result.verdict} ({result.reason})"
        )
        assert "negation" in result.reason.lower() or "prefix" in result.reason.lower()

    def test_total_pair_count_at_least_30(self) -> None:
        total = (
            len(ACCEPT_PAIRS)
            + len(VERIFY_PAIRS)
            + len(REJECT_NUMBERS_PAIRS)
            + len(REJECT_UNIT_INVERSION_PAIRS)
            + len(REJECT_NEGATION_SCOPE_PAIRS)
        )
        assert total >= 30, f"Expected at least 30 test pairs, found {total}"

    def test_normalization_expands_apostrophes_and_strips_punctuation(self) -> None:
        norm = normalize_text("What's the capital of France? Can't wait!")
        assert norm == "what is the capital of france can not wait"

    def test_custom_stopwords_and_safe_words(self) -> None:
        res = check_match(
            "Show me the code snippet",
            "Show the code snippet",
            stopwords={"show", "the"},
            paraphrase_safe_words={"me"},
        )
        assert res.verdict == GuardVerdict.ACCEPT
