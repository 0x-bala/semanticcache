"""The guard must reject questions that are identical except that two words trade places.

A similarity model (embeddings and cross-encoders alike) scores "Flights from Delhi to Mumbai" and
"Flights from Mumbai to Delhi" as near-identical, so a verifier would approve the wrong answer.
"""

from __future__ import annotations

import pytest

from semantic_cache.guard import GuardVerdict, _is_role_swap, check_match

SWAPPED_ROLES = [
    ("Flights from Delhi to Mumbai", "Flights from Mumbai to Delhi"),
    ("Flights from Chennai to Delhi", "Flights from Delhi to Chennai"),
    ("from Pune to Mumbai", "from Mumbai to Pune"),
    ("Did Google acquire YouTube?", "Did YouTube acquire Google?"),
    (
        "How do I transfer money from savings to checking?",
        "How do I transfer money from checking to savings?",
    ),
    ("Is Python faster than Java?", "Is Java faster than Python?"),
    ("Did Brazil beat Germany?", "Did Germany beat Brazil?"),
    ("Train from Chennai to Bangalore", "Train from Bangalore to Chennai"),
    ("How do I convert PDF to Word?", "How do I convert Word to PDF?"),
]


@pytest.mark.parametrize(("cached", "query"), SWAPPED_ROLES)
def test_swapped_roles_are_rejected_not_sent_to_a_verifier(cached: str, query: str) -> None:
    result = check_match(query, cached)
    assert result.verdict == GuardVerdict.REJECT, result.reason


@pytest.mark.parametrize(("cached", "query"), SWAPPED_ROLES)
def test_swapped_roles_are_rejected_in_both_directions(cached: str, query: str) -> None:
    assert check_match(cached, query).verdict == GuardVerdict.REJECT


@pytest.mark.parametrize(
    ("cached", "query"),
    [
        # Rewordings that change the sentence structure are not swaps: the verifier decides.
        ("Who is Apple CEO?", "Who is the CEO of Apple?"),
        ("Is Paris the capital of France?", "Is the capital of France Paris?"),
    ],
)
def test_rewordings_are_still_left_to_the_verifier(cached: str, query: str) -> None:
    result = check_match(query, cached)
    assert result.verdict != GuardVerdict.REJECT, result.reason


def test_identical_questions_are_still_accepted() -> None:
    result = check_match("Flights from Delhi to Mumbai", "Flights from Delhi to Mumbai")
    assert result.verdict == GuardVerdict.ACCEPT


@pytest.mark.parametrize(
    ("new", "cached", "expected"),
    [
        (
            ["flights", "from", "delhi", "to", "mumbai"],
            ["flights", "from", "mumbai", "to", "delhi"],
            True,
        ),
        (["a", "b"], ["b", "a"], True),
        (["a", "b", "c"], ["c", "b", "a"], True),  # ends exchanged
        (["ceo", "of", "apple"], ["apple", "ceo"], False),  # different length
        (["a", "b", "c"], ["b", "c", "a"], False),  # rotation
        (["a", "b", "c", "d"], ["b", "a", "d", "c"], False),  # two exchanges: not a single swap
        (["a", "b", "c"], ["a", "b", "c"], False),  # identical
        (["a", "b"], ["a", "c"], False),  # one position differs
        (["a", "b", "c"], ["a", "x", "y"], False),  # different words
        ([], [], False),
    ],
)
def test_is_role_swap(new: list[str], cached: list[str], expected: bool) -> None:
    assert _is_role_swap(new, cached) is expected
