"""Tests for the per-kind ordering of the numeric-facts guard rule.

Range swaps and unit swaps must still REJECT; a reworded conversion whose
facts appear in a different token order must not be REJECTED by this rule.
"""

from __future__ import annotations

from semantic_cache.guard import GuardVerdict, check_match


def test_range_swap_still_rejects() -> None:
    result = check_match(
        "What is the distance from 5 to 10 km?",
        "What is the distance from 10 to 5 km?",
    )
    assert result.verdict == GuardVerdict.REJECT


def test_unit_swap_still_rejects() -> None:
    result = check_match("Convert 10 km to miles", "Convert 10 miles to km")
    assert result.verdict == GuardVerdict.REJECT


def test_reworded_conversion_not_rejected_by_this_rule() -> None:
    # "How many miles is 10 kilometers?" and "Convert 10 km to miles" have the
    # same facts but in a different token order. The numeric-facts rule must
    # not REJECT them; a later rule (content words) may still VERIFY them.
    result = check_match(
        "How many miles is 10 kilometers?",
        "Convert 10 km to miles",
    )
    assert result.verdict != GuardVerdict.REJECT
    assert not result.reason.startswith("Ordered number/unit facts differ")