"""Hit guard for SemanticCache.

Prevents false-positive semantic cache hits caused by embedding space
blind spots (such as number differences, unit inversions, negation flips,
and different entity slots with identical sentence templates).
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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

# When both queries state an explicit question type, a different wh-word can
# indicate a different requested answer (for example "who" vs "what").
QUESTION_INTENT_WORDS: set[str] = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
}

DEFAULT_STOPWORDS: set[str] = {
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "to", "of", "in", "for", "on", "with", "at", "by", "up", "about",
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
    "wrote",
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


# Treat these as dynamic scopes. The cache layer bypasses storage and retrieval
# for them by default because identical text such as "latest price" can age.
TIME_SENSITIVE_TOKENS: set[str] = {
    "today", "yesterday", "tomorrow", "now", "currently", "current",
    "latest", "recent", "recently", "tonight", "live", "updated",
    "this_week", "this_month", "this_year", "right_now",
}

# Opposites that can occur in highly similar embeddings. Keep this list
# intentionally explicit; do not use broad stemming or inferred antonyms.
ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("enable", "disable"), ("import", "export"), ("encrypt", "decrypt"),
    ("increase", "decrease"), ("upgrade", "downgrade"), ("min", "max"),
    ("minimum", "maximum"), ("asc", "desc"), ("ascending", "descending"),
    ("before", "after"), ("start", "stop"), ("buy", "sell"),
    ("allow", "block"), ("allow", "deny"), ("accept", "reject"),
    ("add", "remove"), ("open", "close"), ("connect", "disconnect"),
    ("install", "uninstall"), ("on", "off"), ("increase", "reduce"),
    ("include", "exclude"), ("enter", "exit"), ("lock", "unlock"),
)

# Exact linear conversions only. Currency, temperature, and storage units are
# deliberately excluded because their conversion rules can be context-dependent.
UNIT_FACTORS: dict[str, tuple[str, Decimal]] = {
    "km": ("length_m", Decimal("1000")),
    "kilometer": ("length_m", Decimal("1000")), "kilometers": ("length_m", Decimal("1000")),
    "kilometre": ("length_m", Decimal("1000")), "kilometres": ("length_m", Decimal("1000")),
    "mile": ("length_m", Decimal("1609.344")), "miles": ("length_m", Decimal("1609.344")),
    "m": ("length_m", Decimal("1")), "meter": ("length_m", Decimal("1")),
    "meters": ("length_m", Decimal("1")),
    "metre": ("length_m", Decimal("1")),
    "metres": ("length_m", Decimal("1")),
    "cm": ("length_m", Decimal("0.01")), "mm": ("length_m", Decimal("0.001")),
    "inch": ("length_m", Decimal("0.0254")), "inches": ("length_m", Decimal("0.0254")),
    "ft": ("length_m", Decimal("0.3048")),
    "foot": ("length_m", Decimal("0.3048")),
    "feet": ("length_m", Decimal("0.3048")),
    "yd": ("length_m", Decimal("0.9144")),
    "yard": ("length_m", Decimal("0.9144")),
    "yards": ("length_m", Decimal("0.9144")),
    "kg": ("mass_kg", Decimal("1")),
    "kilogram": ("mass_kg", Decimal("1")),
    "kilograms": ("mass_kg", Decimal("1")),
    "g": ("mass_kg", Decimal("0.001")),
    "gram": ("mass_kg", Decimal("0.001")),
    "grams": ("mass_kg", Decimal("0.001")),
    "mg": ("mass_kg", Decimal("0.000001")),
    "milligram": ("mass_kg", Decimal("0.000001")),
    "milligrams": ("mass_kg", Decimal("0.000001")),
    "lb": ("mass_kg", Decimal("0.45359237")),
    "lbs": ("mass_kg", Decimal("0.45359237")),
    "pound": ("mass_kg", Decimal("0.45359237")),
    "pounds": ("mass_kg", Decimal("0.45359237")),
    "oz": ("mass_kg", Decimal("0.028349523125")),
    "ounce": ("mass_kg", Decimal("0.028349523125")),
    "ounces": ("mass_kg", Decimal("0.028349523125")),
    "second": ("time_s", Decimal("1")),
    "seconds": ("time_s", Decimal("1")),
    "sec": ("time_s", Decimal("1")),
    "secs": ("time_s", Decimal("1")),
    "minute": ("time_s", Decimal("60")),
    "minutes": ("time_s", Decimal("60")),
    "min": ("time_s", Decimal("60")),
    "mins": ("time_s", Decimal("60")),
    "hour": ("time_s", Decimal("3600")),
    "hours": ("time_s", Decimal("3600")),
    "hr": ("time_s", Decimal("3600")),
    "hrs": ("time_s", Decimal("3600")),
    "day": ("time_s", Decimal("86400")), "days": ("time_s", Decimal("86400")),
    "week": ("time_s", Decimal("604800")), "weeks": ("time_s", Decimal("604800")),
    "liter": ("volume_l", Decimal("1")),
    "liters": ("volume_l", Decimal("1")),
    "litre": ("volume_l", Decimal("1")),
    "litres": ("volume_l", Decimal("1")),
    "ml": ("volume_l", Decimal("0.001")),
}


def is_time_sensitive_query(text: str) -> bool:
    """Return True for relative-time wording whose answer can change quickly."""
    tokens = set(normalize_text(text).split())
    normalized = normalize_text(text)
    relative_phrases = ("this week", "this month", "this year", "right now")
    return bool(tokens & TIME_SENSITIVE_TOKENS) or any(
        phrase in normalized for phrase in relative_phrases
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize query text for guard comparison and embedding lookup.

    1. Lowercase and strip whitespace.
    2. Expand common apostrophe contractions.
    3. Normalize prefixes (non-, un-).
    4. Strip punctuation and collapse whitespace.

    .. note::

       This function strips **all** symbols (``+``, ``#``, ``%``, ``$``,
       ``<``, ``>``, ``=``, …) and is therefore **not suitable** for the
       exact-match key.  Use :func:`exact_match_key` for the exact-match
       tier instead.
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


def exact_match_key(text: str) -> str:
    """Conservative key for the exact-match tier.

    Unlike ``normalize_text``, this deliberately preserves symbols and
    operators because they can change an answer (for example C++ vs C# or
    ``x>5`` vs ``x<5``).

    Steps:
      1. Unicode NFKC normalisation.
      2. Case-fold (locale-independent lower-case).
      3. Collapse whitespace to a single space and strip.
      4. Strip only trailing ``?``, ``.``, and ``!``.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip("?.!").rstrip()


def _significant_symbols(text: str) -> list[str]:
    """Extract operators, currency and symbols that are meaningful in a query."""
    normalized = unicodedata.normalize("NFKC", text)
    symbols: list[str] = []
    for ch in normalized:
        if ch in "+#%$<>=/&*^~|\\@" or unicodedata.category(ch).startswith("S"):
            symbols.append(ch)
    return symbols


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


def _has_antonym_flip(tokens1: list[str], tokens2: list[str]) -> bool:
    set1, set2 = set(tokens1), set(tokens2)
    return any(
        (a in set1 and b in set2) or (b in set1 and a in set2)
        for a, b in ANTONYM_PAIRS
    )


def _directional_markers(tokens: list[str]) -> list[str]:
    markers: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "out" and i + 1 < len(tokens) and tokens[i + 1] == "of":
            markers.append("out_of")
            i += 2
            continue
        if tokens[i] in {
            "from", "to", "into", "onto", "through", "before", "after",
            "above", "below", "over", "under", "between", "versus", "vs",
        }:
            markers.append(tokens[i])
        i += 1
    return markers


def _number_value(token: str) -> Decimal | None:
    if token in NUMBER_WORDS:
        return Decimal(NUMBER_WORDS[token])
    try:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return Decimal(token)
    except InvalidOperation:
        return None
    return None


def _semantic_numeric_facts(tokens: list[str]) -> list[tuple[str, str, str]]:
    """Extract ordered numeric/unit facts, canonicalizing simple equivalent units.

    For explicit conversion questions, only quantities before the first target
    boundary ("to"/"into") are converted. Target units and target numbers remain
    distinct so converting 1 hour to minutes cannot collide with converting it to seconds.
    """
    converter_intent = any(t in {"convert", "conversion", "translate"} for t in tokens)
    target_boundary = None
    if converter_intent:
        for idx, token in enumerate(tokens):
            if token in {"to", "into"}:
                target_boundary = idx
                break

    facts: list[tuple[str, str, str]] = []
    skip: set[int] = set()
    for i, token in enumerate(tokens):
        if i in skip:
            continue
        num = _number_value(token)
        if num is not None:
            unit_idx = i + 1
            can_convert = unit_idx < len(tokens) and tokens[unit_idx] in UNIT_FACTORS
            if target_boundary is not None and i >= target_boundary:
                can_convert = False
            if can_convert:
                unit = tokens[unit_idx]
                dimension, factor = UNIT_FACTORS[unit]
                canonical = (num * factor).normalize()
                facts.append(("quantity", dimension, format(canonical, "f")))
                skip.add(unit_idx)
            else:
                facts.append(("number", "scalar", format(num.normalize(), "f")))
        elif token in CANONICAL_UNITS:
            facts.append(("unit", "unit", CANONICAL_UNITS[token]))
    return facts


def _content_tokens_with_unit_dimensions(tokens: list[str]) -> list[str]:
    """Normalize already-validated number/unit pairs for word-overlap checks."""
    result: list[str] = []
    i = 0
    while i < len(tokens):
        if (
            _number_value(tokens[i]) is not None
            and i + 1 < len(tokens)
            and tokens[i + 1] in UNIT_FACTORS
        ):
            dimension = UNIT_FACTORS[tokens[i + 1]][0]
            result.extend((f"quantity_{dimension}", f"unit_{dimension}"))
            i += 2
            continue
        result.append(UNIT_FACTORS[tokens[i]][0] if tokens[i] in UNIT_FACTORS else tokens[i])
        i += 1
    return result


def _canonicalize_synonyms(
    tokens: list[str], synonym_groups: Iterable[Iterable[str]] | None
) -> list[str]:
    if not synonym_groups:
        return tokens
    mapping: dict[str, str] = {}
    for group in synonym_groups:
        words = sorted({str(word).lower().strip() for word in group if str(word).strip()})
        if len(words) >= 2:
            canonical = words[0]
            for word in words:
                mapping[word] = canonical
    return [mapping.get(token, token) for token in tokens]


# ---------------------------------------------------------------------------
# Public Guard Check
# ---------------------------------------------------------------------------

def check_match(
    new_query: str,
    cached_query: str,
    stopwords: set[str] | None = None,
    paraphrase_safe_words: set[str] | None = None,
    synonym_groups: Iterable[Iterable[str]] | None = None,
) -> GuardResult:
    """Evaluate whether new_query safely matches cached_query.

    Returns:
    - ACCEPT: Difference is only paraphrase-safe words and order is preserved.
    - REJECT: Incompatible numbers, unit order inversion, negation flips, or
      symbol/operator differences.
    - VERIFY: One-sided extra content words (beyond the safe-word set),
      both sides have different content words (slot-value substitution), or
      content words appear in a different order.
    """
    stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    # Relative-time requests are deliberately not treated as reusable semantic hits.
    if is_time_sensitive_query(new_query) or is_time_sensitive_query(cached_query):
        return GuardResult(GuardVerdict.REJECT, "Relative-time query is time-sensitive")
    safe_words = (
        paraphrase_safe_words
        if paraphrase_safe_words is not None
        else DEFAULT_PARAPHRASE_SAFE_WORDS
    )

    # Significant symbols/operators must match; if they differ, reject.
    # Checked BEFORE normalization comparison so stripped symbols are not ignored.
    if _significant_symbols(new_query) != _significant_symbols(cached_query):
        return GuardResult(GuardVerdict.REJECT, "Significant symbols or operators differ")

    norm_new = normalize_text(new_query)
    norm_cached = normalize_text(cached_query)

    # Identical after normalization
    if norm_new == norm_cached:
        return GuardResult(GuardVerdict.ACCEPT, "Exact match after normalization")

    tokens_new = norm_new.split()
    tokens_cached = norm_cached.split()

    # Do not ignore a change in explicit question intent just because wh-words
    # are also stopwords for content overlap. Missing question framing is okay:
    # "Capital city of France?" may match "What is the capital of France?".
    question_new = next((t for t in tokens_new if t in QUESTION_INTENT_WORDS), None)
    question_cached = next((t for t in tokens_cached if t in QUESTION_INTENT_WORDS), None)
    if (
        question_new is not None
        and question_cached is not None
        and question_new != question_cached
    ):
        return GuardResult(
            GuardVerdict.REJECT,
            f"Question intent differs: {question_new!r} vs {question_cached!r}",
        )

    # Ordered number/unit facts catch swapped values and permit only explicit,
    # exact linear conversions such as 1 hour == 60 minutes.
    facts_new = _semantic_numeric_facts(tokens_new)
    facts_cached = _semantic_numeric_facts(tokens_cached)
    if facts_new != facts_cached:
        return GuardResult(
            GuardVerdict.REJECT,
            f"Ordered number/unit facts differ: {facts_new} vs {facts_cached}",
        )

    # Directional markers must keep their order (e.g. from A to B != to B from A).
    directions_new = _directional_markers(tokens_new)
    directions_cached = _directional_markers(tokens_cached)
    if directions_new != directions_cached:
        # A missing marker on one side can be an extra instruction (for example,
        # "summarize this article" vs "summarize this article and translate it
        # to French"). Keep that case in VERIFY so content-difference handling
        # can decide conservatively. Reject only when both queries explicitly
        # specify directional relations and those relations disagree.
        if directions_new and directions_cached:
            return GuardResult(
                GuardVerdict.REJECT,
                f"Directional relation order differs: {directions_new} vs {directions_cached}",
            )
        return GuardResult(
            GuardVerdict.VERIFY,
            f"Directional markers differ: {directions_new} vs {directions_cached}",
        )

    # Rule (c): REJECT if negation or scope words differ or prefix flips exist
    neg_new = {t for t in tokens_new if t in NEGATION_SCOPE_WORDS}
    neg_cached = {t for t in tokens_cached if t in NEGATION_SCOPE_WORDS}
    if neg_new != neg_cached:
        return GuardResult(
            GuardVerdict.REJECT,
            f"Negation or scope words differ: {neg_new} vs {neg_cached}",
        )

    if _has_prefix_flip(tokens_new, tokens_cached) or _has_antonym_flip(
        tokens_new, tokens_cached
    ):
        return GuardResult(
            GuardVerdict.REJECT,
            "Negation or lexical antonym flip detected",
        )

    # Content word analysis for Rules (d) and (e)
    # Remove stopwords and punctuation remnants
    # When a number/unit pair was already validated as an equivalent quantity,
    # compare its unit by dimension (hour/minute both time) rather than spelling.
    # Standalone/target units remain distinguished by the ordered facts check above.
    content_tokens_new = _content_tokens_with_unit_dimensions(tokens_new)
    content_tokens_cached = _content_tokens_with_unit_dimensions(tokens_cached)
    canonical_new = _canonicalize_synonyms(content_tokens_new, synonym_groups)
    canonical_cached = _canonicalize_synonyms(content_tokens_cached, synonym_groups)
    content_new = {t for t in canonical_new if t not in stopwords}
    content_cached = {t for t in canonical_cached if t not in stopwords}

    unmatched_new = (content_new - content_cached) - safe_words
    unmatched_cached = (content_cached - content_new) - safe_words

    # Rule (d): VERIFY if content differs in either direction, unless every
    # addition is explicitly marked safe by the editable safe-word list.
    if unmatched_new and unmatched_cached:
        return GuardResult(
            GuardVerdict.VERIFY,
            f"Different slot values on both sides: {unmatched_new} vs {unmatched_cached}",
        )

    # Rule (d'): one-sided extra content words that are NOT all safe → VERIFY
    if unmatched_new or unmatched_cached:
        return GuardResult(
            GuardVerdict.VERIFY,
            f"One-sided content addition: {unmatched_new | unmatched_cached}",
        )

    # Rule (d''): content words in a different order → VERIFY
    ordered_new = [t for t in canonical_new if t not in stopwords and t not in safe_words]
    ordered_cached = [t for t in canonical_cached if t not in stopwords and t not in safe_words]
    if ordered_new != ordered_cached:
        return GuardResult(GuardVerdict.VERIFY, "Content words appear in different order")

    # Rule (e): ACCEPT if only safe-word changes remain.
    return GuardResult(
        GuardVerdict.ACCEPT,
        "Acceptable paraphrase with one-sided or safe word additions",
    )
