"""Core semantic cache: look up LLM answers by meaning, not exact text.

Design principles
-----------------
- Correctness first: default to a cache MISS when matching is uncertain.
- Privacy: queries and responses are never logged at DEBUG level by default.
  Set ``log_queries=True`` only in development environments.
- Fail-safe: exceptions from the compute function are never cached.
- Concurrency: single-flight coalescing prevents duplicate LLM calls for
  simultaneous identical misses (within one process).
- Backward-compatible: the original get / set / get_or_compute / @cached API
  is fully preserved.

Security notes
--------------
- Namespace is a *partitioning* mechanism, not a security boundary.
- Do not share a SemanticCache instance across tenants without explicit
  namespace segregation.
- Embedding similarity != semantic equivalence.  High threshold values
  (>= 0.90) are strongly recommended for production.
- Set ``enabled=False`` at the call-site for sensitive, personalised,
  or time-critical queries that must never be served from cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import random
import threading
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .embedders import (
    Embedder,
    EmbeddingLRUCache,
    HashingEmbedder,
    is_zero_vector,
    validate_embedding,
)
from .guard import (
    GuardVerdict,
    check_match,
    exact_match_key,
    is_time_sensitive_query,
    normalize_text,
)
from .protocols import CacheStore
from .store import SQLiteStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
_MIN_THRESHOLD = 0.0
_MAX_THRESHOLD = 1.0
_MIN_TTL = 0.0
_MAX_NAMESPACE_LEN = 256


def _validate_threshold(value: float) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"threshold must be a float, got {type(value).__name__}")
    if not (_MIN_THRESHOLD <= value <= _MAX_THRESHOLD):
        raise ValueError(f"threshold must be in [{_MIN_THRESHOLD}, {_MAX_THRESHOLD}], got {value}")
    return float(value)


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    return query


def _validate_query_for_set(query: str) -> str:
    """Stricter validation for set(): reject empty or whitespace-only queries."""
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    if not query.strip():
        raise ValueError("query must not be empty or whitespace-only")
    return query


def _validate_namespace(ns: str) -> str:
    if not isinstance(ns, str) or not ns:
        raise ValueError("namespace must be a non-empty string")
    if len(ns) > _MAX_NAMESPACE_LEN:
        raise ValueError(f"namespace too long: {ns[:40]!r}...")
    return ns


def _detect_embedder_name(embedder: Any) -> str:
    if hasattr(embedder, "_embedder") and embedder._embedder is not None:
        return _detect_embedder_name(embedder._embedder)
    if hasattr(embedder, "model_name") and embedder.model_name:
        return str(embedder.model_name)
    if hasattr(embedder, "model") and embedder.model:
        return str(embedder.model)
    if hasattr(embedder, "name") and embedder.name:
        return str(embedder.name)
    cls_name = embedder.__class__.__name__
    # Embedding algorithms that change their vector representation should
    # expose cache_version. Partitioning by version prevents cosine comparisons
    # between vectors produced by incompatible feature formats.
    if hasattr(embedder, "cache_version"):
        version = str(embedder.cache_version)
        if hasattr(embedder, "dim"):
            return f"{cls_name}:v{version}:{embedder.dim}"
        return f"{cls_name}:v{version}"
    if hasattr(embedder, "dim"):
        return f"{cls_name}:{embedder.dim}"
    return cls_name


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    """Accumulates cache hit/miss statistics.

    ``seconds_saved`` is an *estimate* based on measuring the duration of the
    original compute call.  It does not account for embedding latency.
    """

    hits: int = 0
    misses: int = 0
    errors: int = 0
    guard_rejects: int = 0
    verify_calls: int = 0
    verify_accepts: int = 0
    exact_hits: int = 0
    shadow_evaluations: int = 0
    shadow_matches: int = 0
    shadow_mismatches: int = 0
    shadow_errors: int = 0
    seconds_saved: float = 0.0
    # entry_id -> measured latency (seconds) of the original compute call
    _latency: dict[int, float] = field(default_factory=dict, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Alias for hit_rate (cache-hit rate / queries that reached lookup)."""
        return self.hit_rate

    @property
    def estimated_false_hit_rate(self) -> float:
        """Estimated false-hit rate from shadow-mode evaluations (0.0 if no evaluations).

        Definition (shadow-based, response-comparison): 
        - Numerator: shadow_mismatches (responses differ per comparator; errors excluded)
        - Denominator: shadow_matches + shadow_mismatches (valid shadow evaluations only)
        - Excludes shadow_errors/timeouts. This measures response-difference rate under
          the configured comparator; it is NOT the same as "false hits on non-duplicate queries"
          which depends on ground-truth labels and query pairing.
        """
        valid = self.shadow_matches + self.shadow_mismatches
        if valid == 0:
            return 0.0
        return self.shadow_mismatches / valid

    @property
    def shadow_mismatch_rate(self) -> float:
        """Alias for estimated_false_hit_rate with explicit naming."""
        return self.estimated_false_hit_rate

    @property
    def guard_rejection_rate(self) -> float:
        """Guard rejection rate among candidates that reached guard evaluation."""
        total_guarded = self.guard_rejects + self.verify_calls + self.hits
        if total_guarded == 0:
            return 0.0
        return self.guard_rejects / total_guarded

    @property
    def verification_rate(self) -> float:
        """Rate of candidates that went to verification."""
        total_guarded = self.guard_rejects + self.verify_calls + self.hits
        if total_guarded == 0:
            return 0.0
        return self.verify_calls / total_guarded

    def reset(self) -> None:
        """Reset all counters and the latency map."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.guard_rejects = 0
        self.verify_calls = 0
        self.verify_accepts = 0
        self.exact_hits = 0
        self.shadow_evaluations = 0
        self.shadow_matches = 0
        self.shadow_mismatches = 0
        self.shadow_errors = 0
        self.seconds_saved = 0.0
        self._latency.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the stats with explicit definitions."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "guard_rejects": self.guard_rejects,
            "verify_calls": self.verify_calls,
            "verify_accepts": self.verify_accepts,
            "exact_hits": self.exact_hits,
            "shadow_evaluations": self.shadow_evaluations,
            "shadow_matches": self.shadow_matches,
            "shadow_mismatches": self.shadow_mismatches,
            "shadow_errors": self.shadow_errors,
            "estimated_false_hit_rate": round(self.estimated_false_hit_rate, 4),
            "shadow_mismatch_rate": round(self.shadow_mismatch_rate, 4),
            "hit_rate": round(self.hit_rate, 4),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "guard_rejection_rate": round(self.guard_rejection_rate, 4),
            "verification_rate": round(self.verification_rate, 4),
            "seconds_saved": round(self.seconds_saved, 4),
            "metric_definitions": {
                "cache_hit_rate": {"numerator": "hits", "denominator": "hits + misses"},
                "shadow_mismatch_rate": {
                    "numerator": "shadow_mismatches",
                    "denominator": "shadow_matches + shadow_mismatches",
                    "notes": "Excludes shadow_errors; depends on comparator",
                },
                "guard_rejection_rate": {
                    "numerator": "guard_rejects",
                    "denominator": "guard_rejects + verify_calls + hits",
                    "notes": "Approximate denominator over guard-evaluated candidates",
                },
            },
        }


@dataclass
class CacheResult:
    """Result returned by ``get`` and ``get_or_compute``."""

    response: str
    hit: bool
    score: float = 0.0
    matched_query: str | None = None


# ---------------------------------------------------------------------------
# Callbacks type aliases
# ---------------------------------------------------------------------------
HitCallback = Callable[[CacheResult], None]
MissCallback = Callable[[str], None]
ErrorCallback = Callable[[str, Exception], None]
AsyncCompute = Callable[[str], Awaitable[str]]
VerifierCallback = Callable[[str, str], bool]
ShadowComparator = Callable[[str, str], bool]
CacheableCallback = Callable[[str], bool]


_BLOCKED_PHRASES = (
    "i don't know",
    "the question is unclear",
    "no question was provided",
)


def _normalize_for_blocklist(response: str) -> str:
    """Normalize a response for blocklist comparison.

    Lowercases, replaces the curly apostrophe (\\u2019) with a straight
    quote, and strips trailing punctuation and whitespace.
    """
    normalized = response.lower().replace("\u2019", "'").strip()
    # Strip trailing ".! " so "I don't know." matches "i don't know".
    normalized = normalized.rstrip(".! ")
    return normalized


def _default_is_cacheable(response: str) -> bool:
    """Default predicate: reject empty strings and known refusal phrases.

    Empty responses stay blocked. Legitimate short answers such as "42" or
    "Paris" are still cached. The following responses are never stored:

    - "I don't know"
    - "The question is unclear"
    - "No question was provided"

    Comparison is done with ``startswith`` after normalization, so
    "I don't know the answer." is also blocked.
    """
    if not isinstance(response, str) or not response.strip():
        return False
    normalized = _normalize_for_blocklist(response)
    return not any(normalized.startswith(phrase) for phrase in _BLOCKED_PHRASES)


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """Semantic cache for LLM calls.

    Parameters
    ----------
    embedder:
        Converts query text to a unit-normalised vector.
        Defaults to ``HashingEmbedder`` (offline, no dependencies).
    store:
        Persistence backend.  Defaults to an in-memory ``SQLiteStore``.
    threshold:
        Minimum cosine similarity [0.0, 1.0] to count as a cache hit.
        Higher values reduce false positives.  Recommended: >= 0.85 for
        production, >= 0.90 for sensitive use-cases.
    ttl:
        Entry lifetime in seconds.  ``None`` means entries never expire.
    namespace:
        Default namespace (partition key).  Change per model / system-prompt /
        user-role to prevent answer cross-contamination.
    enabled:
        When ``False`` the cache is completely bypassed — ``get`` always
        returns ``None`` and ``set`` is a no-op.  Useful for A/B testing or
        disabling cache for specific request types.
    log_queries:
        When ``True``, query text is included in DEBUG log messages.
        **Never enable in production** — queries may contain PII.
    on_hit:
        Optional callback invoked on every cache hit.
    on_miss:
        Optional callback invoked on every cache miss.
    on_error:
        Optional callback invoked when a compute function raises.
    guard:
        When ``True`` (default), run semantic hit guard on candidate hits to
        prevent number mismatches, unit inversions, and entity slot substitutions.
    guard_buffer:
        Score range [threshold, threshold + guard_buffer] subject to VERIFY.
        Candidates above this threshold skip VERIFY, but still apply REJECT rules.
        Default is 0.03.
    verifier:
        Optional callable ``verifier(new_query, cached_query) -> bool`` called
        when guard returns VERIFY within the borderline buffer. If None, VERIFY
        is treated as a cache miss.
    exact_match:
        When ``True`` (default), check normalized query hash first to return
        immediate exact hits without calling the embedder.
    embedding_lru_capacity:
        Maximum number of vectors to retain in the embedding LRU cache (default 1000).
    shadow_max_concurrency:
        Maximum number of concurrent background shadow evaluation threads.
        Default is 2.  Set higher for high-throughput services.
    shadow_comparator:
        Optional ``(fresh_response, cached_response) -> bool`` function used
        to compare shadow responses.  Falls back to the verifier, then to
        exact string comparison.
    is_cacheable:
        Optional ``(response: str) -> bool`` predicate that gates whether a
        computed response is stored.  Default: reject empty strings and
        strings shorter than 3 characters.  Use to filter out error/fallback
        phrases such as "I don't know" or "No question was provided."
    allow_question_intent_mismatch:
        When True AND a verifier is configured, the "Question intent differs"
        guard rule (e.g. "who" vs "what") returns VERIFY with
        ``must_verify=True`` instead of REJECT, letting the verifier
        adjudicate.  All other REJECT rules are unchanged.  Has no effect
        when no verifier is configured (REJECT regardless).  Default False.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: CacheStore | None = None,
        threshold: float | None = None,
        ttl: float | None = None,
        namespace: str = "default",
        enabled: bool = True,
        log_queries: bool = False,
        on_hit: HitCallback | None = None,
        on_miss: MissCallback | None = None,
        on_error: ErrorCallback | None = None,
        *,
        guard: bool = True,
        guard_buffer: float = 0.03,
        verifier: VerifierCallback | None = None,
        verify_always: bool = True,
        shadow_rate: float = 0.0,
        shadow_max_concurrency: int = 2,
        shadow_comparator: ShadowComparator | None = None,
        exact_match: bool = True,
        embedding_lru_capacity: int = 1000,
        top_k: int = 1,
        synonym_groups: Iterable[Iterable[str]] | None = None,
        is_cacheable: CacheableCallback | None = None,
        allow_question_intent_mismatch: bool = False,
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._embedder_name: str = _detect_embedder_name(self.embedder)
        _created_store = False
        if store is None:
            self.store = SQLiteStore()
            _created_store = True
        else:
            self.store = store
        self._owns_store = _created_store

        try:
            if threshold is None:
                threshold = float(getattr(self.embedder, "default_threshold", 0.85))
            self.threshold: float = _validate_threshold(threshold)
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                raise ValueError("top_k must be a positive integer")
            if top_k > 500:
                raise ValueError("top_k must not exceed 500")
            self.top_k = top_k
            normalized_groups: list[frozenset[str]] = []
            for group in synonym_groups or ():
                words = frozenset(str(word).strip().lower() for word in group if str(word).strip())
                if len(words) >= 2:
                    normalized_groups.append(words)
            self.synonym_groups = tuple(normalized_groups)
            self.ttl = ttl
            self.namespace: str = _validate_namespace(namespace)
            self.enabled: bool = enabled
            self._log_queries: bool = log_queries
            self._on_hit = on_hit
            self._on_miss = on_miss
            self._on_error = on_error
            self.guard: bool = guard
            self.guard_buffer: float = guard_buffer
            self.verifier: VerifierCallback | None = verifier
            self.verify_always: bool = verify_always
            self.allow_question_intent_mismatch: bool = bool(allow_question_intent_mismatch)
            self.shadow_rate: float = max(0.0, min(1.0, float(shadow_rate)))
            self.shadow_max_concurrency: int = max(1, int(shadow_max_concurrency))
            self.shadow_comparator: ShadowComparator | None = shadow_comparator
            self.exact_match: bool = exact_match
            self.is_cacheable: CacheableCallback = is_cacheable or _default_is_cacheable
            self.allow_question_intent_mismatch: bool = bool(allow_question_intent_mismatch)
            self._embedding_lru = EmbeddingLRUCache(capacity=embedding_lru_capacity)
            self.stats = Stats()
            self._stats_lock = threading.Lock()

            # Shadow mode: bounded thread pool (daemon threads, never block shutdown)
            self._shadow_pool: ThreadPoolExecutor | None = None
            self._shadow_futures: list[Any] = []
            self._shadow_futures_lock = threading.Lock()
            if self.shadow_rate > 0.0:
                self._shadow_pool = ThreadPoolExecutor(
                    max_workers=self.shadow_max_concurrency,
                    thread_name_prefix="semantic-cache-shadow",
                )

            # Single-flight: query -> (lock, result/exception)
            self._inflight: dict[str, list[Any]] = {}
        except BaseException:
            if _created_store and hasattr(self.store, "close"):
                try:
                    self.store.close()
                except Exception:
                    pass
            raise
        self._inflight_lock = threading.Lock()
        # Async single-flight is scoped to an event loop.  A cache instance is
        # normally shared by one ASGI application's loop.
        self._async_inflight: dict[str, asyncio.Future[CacheResult]] = {}

    # ------------------------------------------------------------------
    # Shadow evaluation (background thread)
    # ------------------------------------------------------------------

    def _compare_shadow_responses(self, fresh: str, cached: str) -> bool:
        """Compare fresh vs cached response using the best available comparator."""
        if self.shadow_comparator is not None:
            try:
                return self.shadow_comparator(fresh, cached)
            except Exception:
                pass  # fall through to next method
        if self.verifier is not None:
            try:
                return self.verifier(fresh, cached)
            except Exception:
                return fresh.strip().lower() == cached.strip().lower()
        return fresh.strip() == cached.strip()

    def _shadow_evaluate(
        self,
        compute: Callable[[str], str],
        query: str,
        cached_response: str,
    ) -> None:
        """Run shadow evaluation in the current thread (called from pool)."""
        try:
            fresh_response = compute(query)
            is_match = self._compare_shadow_responses(fresh_response, cached_response)
            with self._stats_lock:
                self.stats.shadow_evaluations += 1
                if is_match:
                    self.stats.shadow_matches += 1
                else:
                    self.stats.shadow_mismatches += 1
        except Exception as exc:
            with self._stats_lock:
                self.stats.shadow_evaluations += 1
                self.stats.shadow_errors += 1
            logger.warning("Shadow mode evaluation failed: %s", exc)

    def _maybe_shadow(
        self,
        compute: Callable[[str], str],
        query: str,
        cached_response: str,
    ) -> None:
        """Submit a shadow evaluation to the thread pool if sampled."""
        if self.shadow_rate <= 0.0 or random.random() >= self.shadow_rate:
            return
        pool = self._shadow_pool
        if pool is None:
            # Create pool lazily (shadow_rate changed after construction)
            self._shadow_pool = ThreadPoolExecutor(
                max_workers=self.shadow_max_concurrency,
                thread_name_prefix="semantic-cache-shadow",
            )
            pool = self._shadow_pool
        try:
            future = pool.submit(self._shadow_evaluate, compute, query, cached_response)
            with self._shadow_futures_lock:
                self._shadow_futures.append(future)
                if len(self._shadow_futures) > 100:
                    self._shadow_futures = [f for f in self._shadow_futures if not f.done()]
        except RuntimeError:
            # Pool shut down — ignore
            pass

    def flush_shadow(self, timeout: float = 2.0) -> None:
        """Wait for pending background shadow evaluations to complete.

        Useful in unit tests and graceful application shutdown.
        """
        with self._shadow_futures_lock:
            pending = list(self._shadow_futures)
        if not pending:
            return
        deadline = time.monotonic() + timeout
        for f in pending:
            remaining = max(0.001, deadline - time.monotonic())
            try:
                f.result(timeout=remaining)
            except Exception:
                pass
        with self._shadow_futures_lock:
            self._shadow_futures = [f for f in self._shadow_futures if not f.done()]

    def close(self, close_store: bool | None = None) -> None:
        """Release cache resources including background shadow thread pool."""
        if self._shadow_pool is not None:
            self._shadow_pool.shutdown(wait=False)
            self._shadow_pool = None
        if close_store is None:
            close_store = getattr(self, "_owns_store", True)
        if close_store and hasattr(self, "store") and hasattr(self.store, "close"):
            self.store.close()

    def __enter__(self) -> SemanticCache:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _verifier_score(self, query_a: str, query_b: str) -> float | None:
        """Return the raw verifier score for a pair, or None if the verifier
        does not expose one (e.g. a plain bool callback)."""
        verifier = self.verifier
        if verifier is None:
            return None
        model = getattr(verifier, "model", None)
        if model is None or not hasattr(model, "predict"):
            return None
        try:
            import numpy as np

            scores = np.asarray(model.predict([[query_a, query_b]])).reshape(-1)
            if scores.size == 1 and np.isfinite(scores[0]):
                return float(scores[0])
        except Exception:
            return None
        return None

    def _verifier_threshold(self) -> float | None:
        """Return the verifier's configured threshold, or None if unknown."""
        verifier = self.verifier
        if verifier is None:
            return None
        threshold = getattr(verifier, "threshold", None)
        if isinstance(threshold, (int, float)):
            return float(threshold)
        return None

    def get(
        self,
        query: str,
        namespace: str | None = None,
        bypass: bool = False,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
    ) -> CacheResult | None:
        """Look up *query* in the cache.

        Returns a ``CacheResult`` on a hit, or ``None`` on a miss.

        Parameters
        ----------
        query:
            The input text to look up.
        namespace:
            Override the instance-level namespace for this call.
        bypass:
            When ``True``, skip the cache and always return ``None``.
        threshold:
            Per-call similarity threshold override.
        guard:
            Per-call toggle for hit guard evaluation (defaults to self.guard).
        verify_always:
            Per-call toggle for whether VERIFY rules apply at all scores.
        """
        _validate_query(query)
        if not self.enabled or bypass:
            return None

        ns = _validate_namespace(namespace or self.namespace)
        eff_threshold = _validate_threshold(threshold) if threshold is not None else self.threshold
        use_guard = self.guard if guard is None else guard
        eff_verify_always = self.verify_always if verify_always is None else verify_always

        # Relative-time prompts are never reused by this cache. This avoids stale
        # results for wording such as "current price" or "latest version".
        if is_time_sensitive_query(query):
            self.stats.misses += 1
            if self._on_miss:
                try:
                    self._on_miss(query)
                except Exception:
                    logger.debug("on_miss callback raised", exc_info=True)
            return None

        norm_query = normalize_text(query)
        em_key = exact_match_key(query)
        query_hash = hashlib.sha256(em_key.encode("utf-8")).hexdigest()

        # Exact match tier: check hash before embedding API call
        if self.exact_match and hasattr(self.store, "find_exact"):
            try:
                exact_entry = self.store.find_exact(ns, query_hash, embedder=self._embedder_name)
                if exact_entry is not None:
                    self.store.record_hit(exact_entry.id)
                    self.stats.hits += 1
                    self.stats.exact_hits += 1
                    saved = self.stats._latency.get(exact_entry.id, 0.0)
                    self.stats.seconds_saved += saved
                    result = CacheResult(
                        response=exact_entry.response,
                        hit=True,
                        score=1.0,
                        matched_query=exact_entry.query,
                    )
                    if self._on_hit:
                        try:
                            self._on_hit(result)
                        except Exception:
                            logger.debug(
                                "on_hit callback raised", exc_info=True
                            )
                    return result
            except Exception as exc:
                logger.warning("Exact match lookup failed: %s", exc)

        # LRU cached embedding
        try:
            vec = self._embedding_lru.get(self._embedder_name, norm_query)
            if vec is None:
                vec = self.embedder.embed(query)
                vec = validate_embedding(vec)
                if not is_zero_vector(vec):
                    self._embedding_lru.put(self._embedder_name, norm_query, vec)
        except Exception as exc:
            logger.warning("Embedder failed during get: %s", exc)
            self.stats.misses += 1
            return None

        if is_zero_vector(vec):
            # Zero vectors produce 0 similarity with everything — treat as miss
            logger.debug("Zero embedding vector — treating as cache miss")
            self.stats.misses += 1
            return None

        try:
            candidates = None
            candidate_search = getattr(self.store, "search_candidates", None)
            if callable(candidate_search):
                try:
                    candidates = candidate_search(
                        ns,
                        vec,
                        embedder=self._embedder_name,
                        top_k=self.top_k,
                        min_similarity=eff_threshold,
                    )
                except TypeError:
                    try:
                        candidates = candidate_search(
                            ns, vec, top_k=self.top_k, min_similarity=eff_threshold
                        )
                    except TypeError:
                        candidates = candidate_search(ns, vec)
            else:
                try:
                    entry = self.store.search(ns, vec, embedder=self._embedder_name)
                except TypeError:
                    entry = self.store.search(ns, vec)
                candidates = [entry] if entry is not None else []
        except Exception as exc:
            logger.warning("Store search failed: %s", exc)
            self.stats.misses += 1
            return None

        # Evaluate candidates from highest to lowest score.
        accepted_entry = None
        for entry in candidates or []:
            if entry is None or entry.score < eff_threshold:
                continue

            if use_guard:
                guard_result = check_match(
                    query,
                    entry.query,
                    synonym_groups=self.synonym_groups,
                    allow_question_intent_mismatch=self.allow_question_intent_mismatch,
                )
                if guard_result.verdict == GuardVerdict.REJECT:
                    logger.debug(
                        "Guard REJECT candidate id=%s: %s", entry.id, guard_result.reason
                    )
                    self.stats.guard_rejects += 1
                    continue
                if guard_result.verdict == GuardVerdict.VERIFY:
                    should_verify = (
                        eff_verify_always
                        or guard_result.must_verify
                        or entry.score < eff_threshold + self.guard_buffer
                    )
                    if should_verify:
                        if self.verifier is None:
                            continue
                        self.stats.verify_calls += 1
                        try:
                            verified = bool(self.verifier(query, entry.query))
                        except Exception as exc:
                            logger.warning("Verifier raised exception: %s", exc)
                            verified = False
                        if not verified:
                            continue
                        self.stats.verify_accepts += 1
            accepted_entry = entry
            break

        if accepted_entry is None:
            self.stats.misses += 1
            if self._on_miss:
                try:
                    self._on_miss(query)
                except Exception:
                    logger.debug("on_miss callback raised", exc_info=True)
            return None

        entry = accepted_entry
        # Hit accepted
        self.store.record_hit(entry.id)
        self.stats.hits += 1
        saved = self.stats._latency.get(entry.id, 0.0)
        self.stats.seconds_saved += saved

        result = CacheResult(
            response=entry.response,
            hit=True,
            score=entry.score,
            matched_query=entry.query,
        )
        if self._log_queries:
            logger.debug(
                "cache.hit ns=%r score=%.4f query=%r matched=%r",
                ns,
                entry.score,
                query,
                entry.query,
            )
        else:
            logger.debug("cache.hit ns=%r score=%.4f", ns, entry.score)

        if self._on_hit:
            try:
                self._on_hit(result)
            except Exception:
                logger.debug("on_hit callback raised", exc_info=True)
        return result

    def explain(
        self,
        query: str,
        namespace: str | None = None,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
    ) -> dict[str, Any]:
        """Describe how *query* would be resolved, without storing anything.

        Reuses the exact search, guard, and verifier path as ``get()`` but
        returns a structured explanation instead of a ``CacheResult``.
        """
        from semantic_cache.guard import GuardVerdict

        _validate_query(query)
        if not self.enabled:
            return {
                "best_candidate": None,
                "threshold": self.threshold,
                "decision": "no_candidates",
                "guard": None,
                "verifier_score": None,
            }

        ns = _validate_namespace(namespace or self.namespace)
        eff_threshold = _validate_threshold(threshold) if threshold is not None else self.threshold
        use_guard = self.guard if guard is None else guard
        eff_verify_always = self.verify_always if verify_always is None else verify_always

        if is_time_sensitive_query(query):
            return {
                "best_candidate": None,
                "threshold": eff_threshold,
                "decision": "no_candidates",
                "guard": None,
                "verifier_score": None,
            }

        norm_query = normalize_text(query)
        em_key = exact_match_key(query)
        query_hash = hashlib.sha256(em_key.encode("utf-8")).hexdigest()

        if self.exact_match and hasattr(self.store, "find_exact"):
            try:
                exact_entry = self.store.find_exact(
                    ns, query_hash, embedder=self._embedder_name
                )
                if exact_entry is not None:
                    return {
                        "best_candidate": {
                            "query": exact_entry.query,
                            "score": 1.0,
                        },
                        "threshold": eff_threshold,
                        "decision": "hit_exact",
                        "guard": None,
                        "verifier_score": None,
                    }
            except Exception as exc:
                logger.warning("Exact match lookup failed: %s", exc)

        try:
            vec = self._embedding_lru.get(self._embedder_name, norm_query)
            if vec is None:
                vec = self.embedder.embed(query)
                vec = validate_embedding(vec)
                if not is_zero_vector(vec):
                    self._embedding_lru.put(self._embedder_name, norm_query, vec)
        except Exception as exc:
            logger.warning("Embedder failed during explain: %s", exc)
            return {
                "best_candidate": None,
                "threshold": eff_threshold,
                "decision": "no_candidates",
                "guard": None,
                "verifier_score": None,
            }

        if is_zero_vector(vec):
            return {
                "best_candidate": None,
                "threshold": eff_threshold,
                "decision": "no_candidates",
                "guard": None,
                "verifier_score": None,
            }

        try:
            candidates = None
            candidate_search = getattr(self.store, "search_candidates", None)
            if callable(candidate_search):
                try:
                    candidates = candidate_search(
                        ns,
                        vec,
                        embedder=self._embedder_name,
                        top_k=self.top_k,
                        min_similarity=eff_threshold,
                    )
                except TypeError:
                    try:
                        candidates = candidate_search(
                            ns, vec, top_k=self.top_k, min_similarity=eff_threshold
                        )
                    except TypeError:
                        candidates = candidate_search(ns, vec)
            else:
                try:
                    entry = self.store.search(ns, vec, embedder=self._embedder_name)
                except TypeError:
                    entry = self.store.search(ns, vec)
                candidates = [entry] if entry is not None else []
        except Exception as exc:
            logger.warning("Store search failed during explain: %s", exc)
            return {
                "best_candidate": None,
                "threshold": eff_threshold,
                "decision": "no_candidates",
                "guard": None,
                "verifier_score": None,
            }

        best_entry = None
        best_guard = None
        best_verifier_score = None
        decision = "no_candidates"
        for entry in candidates or []:
            if entry is None or entry.score < eff_threshold:
                decision = "below_threshold"
                continue

            if use_guard:
                guard_result = check_match(
                    query,
                    entry.query,
                    synonym_groups=self.synonym_groups,
                    allow_question_intent_mismatch=self.allow_question_intent_mismatch,
                )
                best_guard = guard_result
                if guard_result.verdict == GuardVerdict.REJECT:
                    best_entry = entry
                    decision = "guard_reject"
                    continue
                if guard_result.verdict == GuardVerdict.VERIFY:
                    should_verify = (
                        eff_verify_always
                        or guard_result.must_verify
                        or entry.score < eff_threshold + self.guard_buffer
                    )
                    if should_verify:
                        if self.verifier is None:
                            best_entry = entry
                            decision = "verifier_reject"
                            continue
                        try:
                            verifier_score = self._verifier_score(query, entry.query)
                            best_verifier_score = verifier_score
                            threshold = self._verifier_threshold()
                            if verifier_score is not None and threshold is not None:
                                verified = bool(verifier_score >= threshold)
                            else:
                                # Plain bool verifier: no score to expose.
                                verified = bool(self.verifier(query, entry.query))
                        except Exception as exc:
                            logger.warning("Verifier raised exception during explain: %s", exc)
                            verified = False
                        if not verified:
                            best_entry = entry
                            decision = "verifier_reject"
                            continue
                    best_entry = entry
                    decision = "hit_verified"
                    break
                # ACCEPT: guard approved without verification.
                best_entry = entry
                decision = "hit_guard_accept"
                break

        return {
            "best_candidate": (
                {"query": best_entry.query, "score": best_entry.score}
                if best_entry is not None
                else None
            ),
            "threshold": eff_threshold,
            "decision": decision,
            "guard": (
                {
                    "verdict": best_guard.verdict.value,
                    "reason": best_guard.reason,
                }
                if best_guard is not None
                else None
            ),
            "verifier_score": best_verifier_score,
        }

    def set(
        self,
        query: str,
        response: str,
        namespace: str | None = None,
        latency: float = 0.0,
        ttl: float | None = None,
    ) -> None:
        """Store a response for *query*.

        Parameters
        ----------
        query:
            The original query string.
        response:
            The response to cache.  Must be a non-empty string.
        namespace:
            Override the instance-level namespace for this call.
        latency:
            The time (seconds) it took to compute *response*.  Used to estimate
            ``stats.seconds_saved`` on future hits.
        """
        _validate_query_for_set(query)
        if not self.enabled or is_time_sensitive_query(query):
            return

        if not isinstance(response, str) or not response:
            raise ValueError("response must be a non-empty string")

        ns = _validate_namespace(namespace or self.namespace)
        norm_query = normalize_text(query)
        em_key = exact_match_key(query)
        query_hash = hashlib.sha256(em_key.encode("utf-8")).hexdigest()

        try:
            vec = self._embedding_lru.get(self._embedder_name, norm_query)
            if vec is None:
                vec = self.embedder.embed(query)
                vec = validate_embedding(vec)
                if not is_zero_vector(vec):
                    self._embedding_lru.put(self._embedder_name, norm_query, vec)
        except Exception as exc:
            logger.warning("Embedder failed during set — entry not stored: %s", exc)
            return

        if is_zero_vector(vec):
            logger.debug("Zero embedding vector — skipping cache set")
            return

        effective_ttl = self.ttl if ttl is None else ttl
        try:
            entry_id = self.store.add(
                ns,
                query,
                vec,
                response,
                effective_ttl,
                query_hash=query_hash,
                embedder=self._embedder_name,
            )
        except TypeError:
            entry_id = self.store.add(ns, query, vec, response, effective_ttl)

        # Only retain latency for a bounded window to prevent unbounded growth.
        # Entries evicted from the store become unreachable, so we can prune
        # latencies for IDs that are no longer in the store.  For simplicity
        # we cap the dict at 10 000 entries (FIFO drop of oldest keys).
        if len(self.stats._latency) >= 10_000:
            oldest_key = next(iter(self.stats._latency))
            del self.stats._latency[oldest_key]
        self.stats._latency[entry_id] = latency

    def get_or_compute(
        self,
        query: str,
        compute: Callable[[str], str],
        namespace: str | None = None,
        bypass: bool = False,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
        ttl: float | None = None,
    ) -> CacheResult:
        """Return a cached result, or call *compute* and store the result.

        Features
        --------
        - **Single-flight**: if two threads call ``get_or_compute`` with the
          same query simultaneously, only one calls ``compute``.  The other
          waits and receives the same result.  This prevents duplicate LLM calls.
        - **Exception safety**: if ``compute`` raises, the exception propagates
        - **No caching of failed results**: only successful, non-empty strings
          are stored.
        """
        _validate_query(query)
        ns = _validate_namespace(namespace or self.namespace)

        if bypass or not self.enabled:
            try:
                response = compute(query)
                if not isinstance(response, str) or not response:
                    raise ValueError(
                        "compute function must return a non-empty string; "
                        f"got {type(response).__name__!r}"
                    )
            except Exception as exc:
                self.stats.errors += 1
                if self._on_error:
                    try:
                        self._on_error(query, exc)
                    except Exception:
                        pass
                raise
            return CacheResult(response=response, hit=False)

        cached = self.get(
            query,
            ns,
            threshold=threshold,
            guard=guard,
            verify_always=verify_always,
        )
        if cached:
            # Shadow evaluation — dispatched to background thread pool
            self._maybe_shadow(compute, query, cached.response)
            return cached

        flight_key = f"{ns}:{query}"

        with self._inflight_lock:
            if flight_key in self._inflight:
                slot = self._inflight[flight_key]
                am_computing = False
            else:
                slot = [threading.Event(), None]
                self._inflight[flight_key] = slot
                am_computing = True

        if not am_computing:
            slot[0].wait()
            outcome = slot[1]
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, CacheResult):
                return CacheResult(
                    response=outcome.response,
                    hit=False,
                    score=outcome.score,
                    matched_query=outcome.matched_query,
                )
            raise RuntimeError("single-flight slot was empty after event was set")

        # This thread is responsible for computing
        try:
            start = time.perf_counter()
            response = compute(query)
            elapsed = time.perf_counter() - start

            if not isinstance(response, str) or not response:
                raise ValueError(
                    "compute function must return a non-empty string; "
                    f"got {type(response).__name__!r}"
                )

            if not self.is_cacheable(response):
                logger.debug(
                    "Skipping cache set for query %r: response rejected by is_cacheable",
                    query,
                )
                result = CacheResult(response=response, hit=False)
            else:
                try:
                    self.set(query, response, namespace, latency=elapsed, ttl=ttl)
                except ValueError as exc:
                    # set() may reject empty/whitespace queries; skip caching.
                    logger.debug(
                        "Skipping cache set for query %r: %s", query, exc
                    )
                result = CacheResult(response=response, hit=False)
        except Exception as exc:
            self.stats.errors += 1
            if self._on_error:
                try:
                    self._on_error(query, exc)
                except Exception:
                    logger.debug("on_error callback raised", exc_info=True)
            slot[1] = exc
            slot[0].set()
            with self._inflight_lock:
                self._inflight.pop(flight_key, None)
            raise
        else:
            slot[1] = result
            slot[0].set()
            with self._inflight_lock:
                self._inflight.pop(flight_key, None)
            return result

    def cached(
        self,
        namespace: str | None = None,
        bypass: bool = False,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
    ):
        """Decorator for any ``fn(prompt: str) -> str`` (e.g. your LLM call).

        Usage::

            cache = SemanticCache(threshold=0.85)

            @cache.cached(namespace="gpt4:v1")
            def ask_llm(prompt: str) -> str:
                return openai.chat(prompt)
        """
        import functools

        def wrap(fn: Callable[[str], str]) -> Callable[[str], str]:
            @functools.wraps(fn)
            def inner(prompt: str) -> str:
                return self.get_or_compute(
                    prompt,
                    fn,
                    namespace,
                    bypass=bypass,
                    threshold=threshold,
                    guard=guard,
                    verify_always=verify_always,
                ).response

            return inner

        return wrap

    async def async_get_or_compute(
        self,
        query: str,
        compute: AsyncCompute,
        namespace: str | None = None,
        bypass: bool = False,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
        ttl: float | None = None,
    ) -> CacheResult:
        """Async equivalent of :meth:`get_or_compute`.

        ``compute`` must return an awaitable string.  Ordinary calls use
        single-flight coalescing within the current event loop; bypass and
        disabled-cache calls always compute independently and never touch the
        embedder or store.
        """
        _validate_query(query)
        ns = _validate_namespace(namespace or self.namespace)

        async def compute_fresh() -> str:
            try:
                response = compute(query)
                if not inspect.isawaitable(response):
                    raise TypeError("async compute function must return an awaitable")
                response = await response
                if not isinstance(response, str) or not response:
                    raise ValueError(
                        "compute function must return a non-empty string; "
                        f"got {type(response).__name__!r}"
                    )
                return response
            except Exception as exc:
                self.stats.errors += 1
                if self._on_error:
                    try:
                        self._on_error(query, exc)
                    except Exception:
                        pass
                raise

        if bypass or not self.enabled:
            return CacheResult(response=await compute_fresh(), hit=False)

        cached = await asyncio.to_thread(
            self.get, query, ns, bypass, threshold, guard, verify_always
        )
        if cached:
            # Shadow evaluation — async compute wrapped in a background task
            if self.shadow_rate > 0.0 and random.random() < self.shadow_rate:

                async def _async_shadow() -> None:
                    try:
                        fresh_response = await compute_fresh()
                        is_match = self._compare_shadow_responses(fresh_response, cached.response)
                        with self._stats_lock:
                            self.stats.shadow_evaluations += 1
                            if is_match:
                                self.stats.shadow_matches += 1
                            else:
                                self.stats.shadow_mismatches += 1
                    except Exception as exc:
                        with self._stats_lock:
                            self.stats.shadow_evaluations += 1
                            self.stats.shadow_errors += 1
                        logger.warning("Shadow mode async evaluation failed: %s", exc)

                asyncio.ensure_future(_async_shadow())
            return cached

        key = f"{ns}:{query}"
        future = self._async_inflight.get(key)
        if future is not None:
            outcome = await asyncio.shield(future)
            return CacheResult(response=outcome.response, hit=False)

        future = asyncio.get_running_loop().create_future()
        self._async_inflight[key] = future
        try:
            start = time.perf_counter()
            response = await compute_fresh()
            elapsed = time.perf_counter() - start

            if not self.is_cacheable(response):
                logger.debug(
                    "Skipping async cache set for query %r: response rejected by is_cacheable",
                    query,
                )
                result = CacheResult(response=response, hit=False)
                future.set_result(result)
                return result

            try:
                await asyncio.to_thread(self.set, query, response, ns, elapsed, ttl)
            except ValueError as exc:
                logger.debug(
                    "Skipping async cache set for query %r: %s", query, exc
                )
            result = CacheResult(response=response, hit=False)
            future.set_result(result)
            return result
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                # Retrieving the exception prevents an unobserved-future warning
                # when there are no waiters.
                future.add_done_callback(lambda completed: completed.exception())
            raise
        finally:
            self._async_inflight.pop(key, None)

    def async_cached(
        self,
        namespace: str | None = None,
        bypass: bool = False,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
    ):
        """Decorator for ``async def fn(prompt: str) -> str`` functions."""
        import functools

        def wrap(fn: AsyncCompute) -> AsyncCompute:
            @functools.wraps(fn)
            async def inner(prompt: str) -> str:
                return (
                    await self.async_get_or_compute(
                        prompt,
                        fn,
                        namespace,
                        bypass=bypass,
                        threshold=threshold,
                        guard=guard,
                        verify_always=verify_always,
                    )
                ).response

            return inner

        return wrap

    # ------------------------------------------------------------------
    # Async get / set / ask API
    # ------------------------------------------------------------------

    async def aget(
        self,
        query: str,
        namespace: str | None = None,
        bypass: bool = False,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
    ) -> CacheResult | None:
        """Asynchronously look up *query* in the cache."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get(
                query,
                namespace=namespace,
                bypass=bypass,
                threshold=threshold,
                guard=guard,
                verify_always=verify_always,
            ),
        )

    async def aset(
        self,
        query: str,
        response: str,
        ttl: float | None = None,
        namespace: str | None = None,
    ) -> None:
        """Asynchronously store *query* -> *response* in the cache."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.set(
                query,
                response,
                ttl=ttl,
                namespace=namespace,
            ),
        )

    async def aask(
        self,
        prompt: str,
        llm: Callable[[str], str] | AsyncCompute,
        namespace: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously ask a prompt, returning cached answer or computing with llm."""
        if inspect.iscoroutinefunction(llm):
            res = await self.async_get_or_compute(prompt, llm, namespace=namespace, **kwargs)
            return res.response
        else:
            async def _async_compute(p: str) -> str:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: llm(p))

            res = await self.async_get_or_compute(
                prompt, _async_compute, namespace=namespace, **kwargs
            )
            return res.response

    # ------------------------------------------------------------------
    # Streaming response caching
    # ------------------------------------------------------------------

    def get_or_compute_stream(
        self,
        query: str,
        compute_stream: Callable[[str], Iterable[str]],
        namespace: str | None = None,
        ttl: float | None = None,
        bypass: bool = False,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
    ) -> Iterator[str]:
        """Stream cached response chunks or compute stream and cache full response."""
        ns = _validate_namespace(namespace or self.namespace)
        if self.enabled and not bypass:
            cached = self.get(
                query,
                namespace=ns,
                threshold=threshold,
                guard=guard,
                verify_always=verify_always,
            )
            if cached is not None and cached.hit:
                yield cached.response
                return

        chunks: list[str] = []
        try:
            stream = compute_stream(query)
            for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            self.stats.errors += 1
            if self._on_error:
                try:
                    self._on_error(query, exc)
                except Exception:
                    pass
            raise

        full_response = "".join(chunks)
        if self.enabled and not bypass and full_response:
            try:
                self.set(query, full_response, ttl=ttl, namespace=ns)
            except Exception as exc:
                logger.warning("Failed to cache streaming response: %s", exc)

    async def async_get_or_compute_stream(
        self,
        query: str,
        compute_stream: Callable[[str], AsyncIterable[str]],
        namespace: str | None = None,
        ttl: float | None = None,
        bypass: bool = False,
        threshold: float | None = None,
        guard: bool | None = None,
        verify_always: bool | None = None,
    ) -> AsyncIterator[str]:
        """Async stream cached response chunks or compute async stream and cache full response."""
        ns = _validate_namespace(namespace or self.namespace)
        if self.enabled and not bypass:
            cached = await self.aget(
                query,
                namespace=ns,
                threshold=threshold,
                guard=guard,
                verify_always=verify_always,
            )
            if cached is not None and cached.hit:
                yield cached.response
                return

        chunks: list[str] = []
        try:
            stream = compute_stream(query)
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            self.stats.errors += 1
            if self._on_error:
                try:
                    self._on_error(query, exc)
                except Exception:
                    pass
            raise

        full_response = "".join(chunks)
        if self.enabled and not bypass and full_response:
            try:
                await self.aset(query, full_response, ttl=ttl, namespace=ns)
            except Exception as exc:
                logger.warning("Failed to cache streaming response: %s", exc)

    # ------------------------------------------------------------------
    # Management helpers
    # ------------------------------------------------------------------

    def clear(self, namespace: str | None = None) -> None:
        """Clear all entries for *namespace* (or all if None)."""
        self.store.clear(namespace)

    def delete(self, query: str, namespace: str | None = None) -> int:
        """Delete entries that exactly match *query*.  Returns deleted count."""
        ns = _validate_namespace(namespace or self.namespace)
        return self.store.delete_by_query(ns, query)

    def invalidate_expired(self) -> int:
        """Delete all expired entries.  Returns deleted count."""
        return self.store.invalidate_expired()

    def get_stats(self) -> dict:
        """Return a JSON-serialisable snapshot of the in-process statistics."""
        return self.stats.as_dict()

    def reset_stats(self) -> None:
        """Reset all in-process hit/miss/error counters."""
        self.stats.reset()

    def namespace_stats(self, namespace: str | None = None) -> dict:
        """Return per-namespace entry counts from the store."""
        ns = _validate_namespace(namespace or self.namespace)
        return self.store.namespace_stats(ns)


# ---------------------------------------------------------------------------
# Standalone @cached decorator
# ---------------------------------------------------------------------------

_DEFAULT_GLOBAL_CACHE: SemanticCache | None = None


def _get_default_cache() -> SemanticCache:
    global _DEFAULT_GLOBAL_CACHE
    if _DEFAULT_GLOBAL_CACHE is None:
        _DEFAULT_GLOBAL_CACHE = SemanticCache()
    return _DEFAULT_GLOBAL_CACHE


def cached(
    cache: SemanticCache | None = None,
    namespace: str | None = None,
    ttl: float | None = None,
    threshold: float | None = None,
    guard: bool | None = None,
    verify_always: bool | None = None,
    key_fn: Callable[..., str] | None = None,
):
    """Decorator to cache results of sync or async functions using semantic cache.

    Can decorate either:
        @cached
        def ask_llm(prompt: str) -> str: ...

        @cached(cache=my_cache, ttl=300)
        async def ask_llm_async(prompt: str) -> str: ...
    """
    import functools

    def decorator(fn: Callable) -> Callable:
        target_cache = cache if cache is not None else _get_default_cache()

        try:
            _params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            _params = []
        _skip_first = bool(_params) and _params[0] in ("self", "cls")

        def extract_query(
            args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[str, bool, dict[str, Any], str | None]:
            """Return (query, bypass, call_kwargs, namespace).

            Every argument other than the prompt (model, temperature, ...) is folded
            into the namespace so calls that differ only in those arguments can never
            share a cached answer.
            """
            call_kwargs = dict(kwargs)
            bypass = bool(call_kwargs.pop("bypass", False))
            if key_fn is not None:
                q = str(key_fn(*args, **call_kwargs))
                extras: list[Any] = []
            else:
                pos = list(args[1:] if _skip_first else args)
                rest = dict(call_kwargs)
                if "prompt" in rest:
                    q = str(rest.pop("prompt"))
                elif "query" in rest:
                    q = str(rest.pop("query"))
                elif pos:
                    q = str(pos.pop(0))
                else:
                    raise ValueError("Could not extract query string for @cached function")
                extras = [pos, sorted(rest.items())]
            ns = namespace
            if extras and (extras[0] or extras[1]):
                digest = hashlib.sha256(repr(extras).encode("utf-8")).hexdigest()[:16]
                ns = f"{namespace or target_cache.namespace}:{digest}"
            return q, bypass, call_kwargs, ns

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                q, bypass, call_kwargs, ns = extract_query(args, kwargs)
                res = await target_cache.async_get_or_compute(
                    q,
                    lambda _: fn(*args, **call_kwargs),
                    namespace=ns,
                    ttl=ttl,
                    bypass=bypass,
                    threshold=threshold,
                    guard=guard,
                    verify_always=verify_always,
                )
                return res.response

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            q, bypass, call_kwargs, ns = extract_query(args, kwargs)
            res = target_cache.get_or_compute(
                q,
                lambda _: fn(*args, **call_kwargs),
                namespace=ns,
                ttl=ttl,
                bypass=bypass,
                threshold=threshold,
                guard=guard,
                verify_always=verify_always,
            )
            return res.response

        return sync_wrapper

    # If called without arguments: @cached
    if callable(cache):
        actual_fn = cache
        cache = None
        return decorator(actual_fn)

    return decorator

