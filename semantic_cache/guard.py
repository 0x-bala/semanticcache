"""Hit guard for SemanticCache.

Prevents false-positive semantic cache hits caused by embedding space
blind spots (such as number differences, unit inversions, negation flips,
and different entity slots with identical sentence templates).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class GuardVerdict(str, Enum):
    """Verdict returned by check_match."""
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    VERIFY = "VERIFY"


@dataclass(frozen=True)
class GuardResult:
    """Outcome of guard evaluation."""
    verdict: GuardVerdict
    reason: str


# ---------------------------------------------------------------------------
# Dictionaries & Configuration
# ---------------------------------------------------------------------------

NUMBER_WORDS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
}

CANONICAL_UNITS: dict[str, str] = {
    # Length / Distance
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "mile": "mile",
    "miles": "mile",
    "m": "meter",
    "meter": "meter",
    "meters": "meter",
    "metre": "meter",
    "metres": "meter",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "inch": "inch",
    "inches": "inch",
    "ft": "foot",
    "foot": "foot",
    "feet": "foot",
    "yd": "yard",
    "yard": "yard",
    "yards": "yard",
    # Mass / Weight
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    "oz": "ounce",
    "ounce": "ounce",
    "ounces": "ounce",
    "ton": "ton",
    "tons": "ton",
    # Temperature
    "celsius": "celsius",
    "fahrenheit": "fahrenheit",
    "kelvin": "kelvin",
    # Currency
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "cent": "cent",
    "cents": "cent",
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "gbp": "gbp",
    "inr": "inr",
    "rupee": "inr",
    "rupees": "inr",
    "jpy": "jpy",
    "yen": "jpy",
    # Time
    "sec": "second",
    "secs": "second",
    "second": "second",
    "seconds": "second",
    "min": "minute",
    "mins": "minute",
    "minute": "minute",
    "minutes": "minute",
    "hr": "hour",
    "hrs": "hour",
    "hour": "hour",
    "hours": "hour",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    # Speed
    "mph": "mph",
    "kph": "kph",
    "kmh": "kph",
    # Data
    "byte": "byte",
    "bytes": "byte",
    "bit": "bit",
    "bits": "bit",
    "kb": "kb",
    "mb": "mb",
    "gb": "gb",
    "tb": "tb",
    # Volume
    "liter": "liter",
    "liters": "liter",
    "litre": "liter",
    "litres": "liter",
    "gallon": "gallon",
    "gallons": "gallon",
    "ml": "ml",
}

NEGATION_SCOPE_WORDS: set[str] = {
    "not",
    "no",
    "never",
    "without",
    "except",
    "only",
    "unless",
    "neither",
    "nor",
    "none",
}

DEFAULT_STOPWORDS: set[str] = {
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about",
    "into", "over", "after",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "it", "its", "i", "me", "my", "myself", "we", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "they", "them", "their", "theirs",
    "how", "why", "when", "where",
}

DEFAULT_PARAPHRASE_SAFE_WORDS: set[str] = {
    "please", "kindly", "tell", "give", "show", "help", "me", "us",
    "play", "city", "book", "film", "movie", "song", "define", "explain",
    "describe", "mean", "meaning", "definition", "query", "question",
}

CONTRACTIONS: list[tuple[str, str]] = [
    (r"\bwhat's\b", "what is"),
    (r"\bhow's\b", "how is"),
    (r"\bwho's\b", "who is"),
    (r"\bwhere's\b", "where is"),
    (r"\bthere's\b", "there is"),
    (r"\bthat's\b", "that is"),
    (r"\bit's\b", "it is"),
    (r"\bcan't\b", "can not"),
    (r"\bwon't\b", "will not"),
    (r"\bdon't\b", "do not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bdidn't\b", "did not"),
    (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"),
    (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"),
    (r"\bhasn't\b", "has not"),
    (r"\bhaven't\b", "have not"),
    (r"\bhadn't\b", "had not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bwouldn't\b", "would not"),
    (r"\bcouldn't\b", "could not"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'ve\b", " have"),
    (r"'ll\b", " will"),
    (r"'d\b", " would"),
    (r"'m\b", " am"),
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize query text for guard comparison and exact matching.

    1. Lowercase and strip whitespace.
    2. Expand common apostrophe contractions.
    3. Normalize prefixes (non-, un-).
    4. Strip punctuation and collapse whitespace.
    """
    text = text.lower().strip()
    for pattern, replacement in CONTRACTIONS:
        text = re.sub(pattern, replacement, text)

    # Normalize 'non-word' -> 'non_word' and 'un-word' -> 'un_word' so prefix is clear
    text = re.sub(r"\bnon-(\w+)\b", r"non_\1", text)
    text = re.sub(r"\bun-(\w+)\b", r"un_\1", text)

    # Replace punctuation (except underscores) with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Helper Extractors
# ---------------------------------------------------------------------------

def _extract_numbers(tokens: list[str]) -> list[str]:
    """Extract list of numbers (digits and number words) in appearance order."""
    numbers: list[str] = []
    for token in tokens:
        if token in NUMBER_WORDS:
            numbers.append(NUMBER_WORDS[token])
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            numbers.append(str(float(token) if "." in token else int(token)))
    return numbers


def _extract_numbers_and_units(tokens: list[str]) -> list[str]:
    """Extract sequence of numbers and canonical units in appearance order."""
    seq: list[str] = []
    for token in tokens:
        if token in NUMBER_WORDS:
            seq.append(NUMBER_WORDS[token])
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            seq.append(str(float(token) if "." in token else int(token)))
        elif token in CANONICAL_UNITS:
            seq.append(CANONICAL_UNITS[token])
    return seq


def _has_prefix_flip(tokens1: list[str], tokens2: list[str]) -> bool:
    """Check if one token list has 'non_X', 'nonX', 'un_X', or 'unX' while the other has 'X'."""
    set1 = set(tokens1)
    set2 = set(tokens2)

    prefixes = ("non_", "un_", "non", "un")
    for t in set1:
        for p in prefixes:
            if t.startswith(p):
                base = t[len(p):]
                if len(base) >= 3 and base in set2:
                    return True
    for t in set2:
        for p in prefixes:
            if t.startswith(p):
                base = t[len(p):]
                if len(base) >= 3 and base in set1:
                    return True
    return False


# ---------------------------------------------------------------------------
# Public Guard Check
# ---------------------------------------------------------------------------

def check_match(
    new_query: str,
    cached_query: str,
    stopwords: set[str] | None = None,
    paraphrase_safe_words: set[str] | None = None,
) -> GuardResult:
    """Evaluate whether new_query safely matches cached_query.

    Returns:
    - ACCEPT: Difference is only one-sided extra words or paraphrase-safe words.
    - REJECT: Incompatible numbers, unit order inversion, or negation flips.
    - VERIFY: Both queries have different content words (slot value substitution).
    """
    stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    safe_words = (
        paraphrase_safe_words
        if paraphrase_safe_words is not None
        else DEFAULT_PARAPHRASE_SAFE_WORDS
    )

    norm_new = normalize_text(new_query)
    norm_cached = normalize_text(cached_query)

    # Identical after normalization
    if norm_new == norm_cached:
        return GuardResult(GuardVerdict.ACCEPT, "Exact match after normalization")

    tokens_new = norm_new.split()
    tokens_cached = norm_cached.split()

    # Rule (a): REJECT if numbers differ
    nums_new = _extract_numbers(tokens_new)
    nums_cached = _extract_numbers(tokens_cached)
    if nums_new != nums_cached:
        return GuardResult(
            GuardVerdict.REJECT,
            f"Numbers differ: {nums_new} vs {nums_cached}",
        )

    # Rule (b): REJECT if sequence of numbers and units differs in order
    units_seq_new = _extract_numbers_and_units(tokens_new)
    units_seq_cached = _extract_numbers_and_units(tokens_cached)
    if units_seq_new != units_seq_cached:
        # Check if the same elements exist in a different order
        if sorted(units_seq_new) == sorted(units_seq_cached) and units_seq_new != units_seq_cached:
            return GuardResult(
                GuardVerdict.REJECT,
                f"Numbers and units inverted: {units_seq_new} vs {units_seq_cached}",
            )
        # Or different units entirely
        return GuardResult(
            GuardVerdict.REJECT,
            f"Units/numbers sequence mismatch: {units_seq_new} vs {units_seq_cached}",
        )

    # Rule (c): REJECT if negation or scope words differ or prefix flips exist
    neg_new = {t for t in tokens_new if t in NEGATION_SCOPE_WORDS}
    neg_cached = {t for t in tokens_cached if t in NEGATION_SCOPE_WORDS}
    if neg_new != neg_cached:
        return GuardResult(
            GuardVerdict.REJECT,
            f"Negation or scope words differ: {neg_new} vs {neg_cached}",
        )

    if _has_prefix_flip(tokens_new, tokens_cached):
        return GuardResult(
            GuardVerdict.REJECT,
            "Negation or antonym prefix flip detected",
        )

    # Content word analysis for Rules (d) and (e)
    # Remove stopwords and punctuation remnants
    content_new = {t for t in tokens_new if t not in stopwords}
    content_cached = {t for t in tokens_cached if t not in stopwords}

    unmatched_new = (content_new - content_cached) - safe_words
    unmatched_cached = (content_cached - content_new) - safe_words

    # Rule (d): VERIFY if both queries have unmatched content words on BOTH sides
    if unmatched_new and unmatched_cached:
        return GuardResult(
            GuardVerdict.VERIFY,
            f"Different slot values on both sides: {unmatched_new} vs {unmatched_cached}",
        )

    # Rule (e): ACCEPT if difference is one-sided only or empty
    return GuardResult(
        GuardVerdict.ACCEPT,
        "Acceptable paraphrase with one-sided or safe word additions",
    )
