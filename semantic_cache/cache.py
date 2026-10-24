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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .embedders import (
    Embedder,
    EmbeddingLRUCache,
    HashingEmbedder,
    is_zero_vector,
    validate_embedding,
)
from .guard import GuardVerdict, check_match, normalize_text
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
        raise ValueError(
            f"threshold must be in [{_MIN_THRESHOLD}, {_MAX_THRESHOLD}], got {value}"
        )
    return float(value)


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
    seconds_saved: float = 0.0
    # entry_id -> measured latency (seconds) of the original compute call
    _latency: dict[int, float] = field(default_factory=dict, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def estimated_false_hit_rate(self) -> float:
        """Estimated false-hit rate from shadow-mode evaluations (0.0 if no evaluations)."""
        if self.shadow_evaluations == 0:
            return 0.0
        return self.shadow_mismatches / self.shadow_evaluations

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
        self.seconds_saved = 0.0
        self._latency.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the stats."""
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
            "estimated_false_hit_rate": round(self.estimated_false_hit_rate, 4),
            "seconds_saved": round(self.seconds_saved, 4),
            "hit_rate": round(self.hit_rate, 4),
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
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: CacheStore | None = None,
        threshold: float = 0.85,
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
        exact_match: bool = True,
        embedding_lru_capacity: int = 1000,
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._embedder_name: str = _detect_embedder_name(self.embedder)
        self.store: CacheStore = store or SQLiteStore()
        self.threshold: float = _validate_threshold(threshold)
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
        self.shadow_rate: float = max(0.0, min(1.0, float(shadow_rate)))
        self.exact_match: bool = exact_match
        self._embedding_lru = EmbeddingLRUCache(capacity=embedding_lru_capacity)
        self.stats = Stats()
        self._stats_lock = threading.Lock()

        # Single-flight: query -> (lock, result/exception)
        self._inflight: dict[str, list[Any]] = {}
        self._inflight_lock = threading.Lock()
        # Async single-flight is scoped to an event loop.  A cache instance is
        # normally shared by one ASGI application's loop.
        self._async_inflight: dict[str, asyncio.Future[CacheResult]] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

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
        if not self.enabled or bypass:
            return None

        ns = _validate_namespace(namespace or self.namespace)
        eff_threshold = _validate_threshold(threshold) if threshold is not None else self.threshold
        use_guard = self.guard if guard is None else guard
        eff_verify_always = self.verify_always if verify_always is None else verify_always

        norm_query = normalize_text(query)
        query_hash = hashlib.sha256(norm_query.encode("utf-8")).hexdigest()

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
                            pass
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
            if hasattr(self.store, "search"):
                try:
                    entry = self.store.search(ns, vec, embedder=self._embedder_name)
                except TypeError:
                    entry = self.store.search(ns, vec)
            else:
                entry = self.store.search(ns, vec)
        except Exception as exc:
            logger.warning("Store search failed: %s", exc)
            self.stats.misses += 1
            return None

        if entry is None or entry.score < eff_threshold:
            self.stats.misses += 1
            if self._on_miss:
                try:
                    self._on_miss(query)
                except Exception:
                    pass
            return None

        # Guard evaluation
        if use_guard:
            guard_result = check_match(query, entry.query)
            if guard_result.verdict == GuardVerdict.REJECT:
                logger.debug("Guard REJECT: %s", guard_result.reason)
                self.stats.guard_rejects += 1
                self.stats.misses += 1
                if self._on_miss:
                    try:
                        self._on_miss(query)
                    except Exception:
                        pass
                return None
            elif guard_result.verdict == GuardVerdict.VERIFY:
                should_verify = eff_verify_always or (
                    entry.score < eff_threshold + self.guard_buffer
                )
                if should_verify:
                    if self.verifier is not None:
                        self.stats.verify_calls += 1
                        try:
                            verified = self.verifier(query, entry.query)
                        except Exception as exc:
                            logger.warning("Verifier raised exception: %s", exc)
                            verified = False

                        if not verified:
                            self.stats.misses += 1
                            if self._on_miss:
                                try:
                                    self._on_miss(query)
                                except Exception:
                                    pass
                            return None
                        self.stats.verify_accepts += 1
                    else:
                        # If no verifier configured, treat VERIFY as a miss
                        self.stats.misses += 1
                        if self._on_miss:
                            try:
                                self._on_miss(query)
                            except Exception:
                                pass
                        return None

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
                ns, entry.score, query, entry.query,
            )
        else:
            logger.debug("cache.hit ns=%r score=%.4f", ns, entry.score)

        if self._on_hit:
            try:
                self._on_hit(result)
            except Exception:
                pass
        return result

    def set(
        self,
        query: str,
        response: str,
        namespace: str | None = None,
        latency: float = 0.0,
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
        if not self.enabled:
            return

        if not isinstance(response, str) or not response:
            raise ValueError("response must be a non-empty string")

        ns = _validate_namespace(namespace or self.namespace)
        norm_query = normalize_text(query)
        query_hash = hashlib.sha256(norm_query.encode("utf-8")).hexdigest()

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

        try:
            entry_id = self.store.add(
                ns,
                query,
                vec,
                response,
                self.ttl,
                query_hash=query_hash,
                embedder=self._embedder_name,
            )
        except TypeError:
            entry_id = self.store.add(ns, query, vec, response, self.ttl)

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
            # Shadow evaluation
            if self.shadow_rate > 0.0 and random.random() < self.shadow_rate:
                try:
                    fresh_response = compute(query)
                    is_match = False
                    if self.verifier is not None:
                        try:
                            is_match = self.verifier(fresh_response, cached.response)
                        except Exception:
                            is_match = (
                                fresh_response.strip().lower() == cached.response.strip().lower()
                            )
                    else:
                        is_match = (fresh_response.strip() == cached.response.strip())

                    with self._stats_lock:
                        self.stats.shadow_evaluations += 1
                        if is_match:
                            self.stats.shadow_matches += 1
                        else:
                            self.stats.shadow_mismatches += 1
                except Exception as exc:
                    logger.warning("Shadow mode evaluation failed: %s", exc)
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

            self.set(query, response, namespace, latency=elapsed)
            result = CacheResult(response=response, hit=False)
        except Exception as exc:
            self.stats.errors += 1
            if self._on_error:
                try:
                    self._on_error(query, exc)
                except Exception:
                    pass
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
    ) -> CacheResult:
        """Async equivalent of :meth:`get_or_compute`.

        ``compute`` must return an awaitable string.  Ordinary calls use
        single-flight coalescing within the current event loop; bypass and
        disabled-cache calls always compute independently and never touch the
        embedder or store.
        """
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
            if self.shadow_rate > 0.0 and random.random() < self.shadow_rate:
                try:
                    fresh_response = await compute_fresh()
                    is_match = False
                    if self.verifier is not None:
                        try:
                            is_match = self.verifier(fresh_response, cached.response)
                        except Exception:
                            is_match = (
                                fresh_response.strip().lower() == cached.response.strip().lower()
                            )
                    else:
                        is_match = (fresh_response.strip() == cached.response.strip())

                    with self._stats_lock:
                        self.stats.shadow_evaluations += 1
                        if is_match:
                            self.stats.shadow_matches += 1
                        else:
                            self.stats.shadow_mismatches += 1
                except Exception as exc:
                    logger.warning("Shadow mode async evaluation failed: %s", exc)
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
            await asyncio.to_thread(self.set, query, response, ns, elapsed)
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
                return (await self.async_get_or_compute(
                    prompt,
                    fn,
                    namespace,
                    bypass=bypass,
                    threshold=threshold,
                    guard=guard,
                    verify_always=verify_always,
                )).response

            return inner
        return wrap

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
