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
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .embedders import Embedder, HashingEmbedder, is_zero_vector
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
    seconds_saved: float = 0.0
    # entry_id -> measured latency (seconds) of the original compute call
    _latency: dict[int, float] = field(default_factory=dict, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset(self) -> None:
        """Reset all counters and the latency map."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.seconds_saved = 0.0
        self._latency.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the stats."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
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
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.store: CacheStore = store or SQLiteStore()
        self.threshold: float = _validate_threshold(threshold)
        self.ttl = ttl
        self.namespace: str = _validate_namespace(namespace)
        self.enabled: bool = enabled
        self._log_queries: bool = log_queries
        self._on_hit = on_hit
        self._on_miss = on_miss
        self._on_error = on_error
        self.stats = Stats()

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
        """
        if not self.enabled or bypass:
            return None

        ns = _validate_namespace(namespace or self.namespace)

        try:
            vec = self.embedder.embed(query)
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
            entry = self.store.search(ns, vec)
        except Exception as exc:
            logger.warning("Store search failed: %s", exc)
            self.stats.misses += 1
            return None

        if entry is not None and entry.score >= self.threshold:
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

        self.stats.misses += 1
        if self._on_miss:
            try:
                self._on_miss(query)
            except Exception:
                pass
        return None

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

        try:
            vec = self.embedder.embed(query)
        except Exception as exc:
            logger.warning("Embedder failed during set — entry not stored: %s", exc)
            return

        if is_zero_vector(vec):
            logger.debug("Zero embedding vector — skipping cache set")
            return

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
    ) -> CacheResult:
        """Return a cached result, or call *compute* and store the result.

        Features
        --------
        - **Single-flight**: if two threads call ``get_or_compute`` with the
          same query simultaneously, only one calls ``compute``.  The other
          waits and receives the same result.  This prevents duplicate LLM calls.
        - **Exception safety**: if ``compute`` raises, the exception propagates
          to the caller, nothing is cached, and the event is set so all waiting
          callers are unblocked and re-raise the same exception.
        - **No caching of failed results**: only successful, non-empty strings
          are stored.

        When ``bypass=True`` (or the cache is globally disabled), the cache is
        not consulted or written to.  Each bypass call computes independently;
        it deliberately does not participate in single-flight coalescing.

        Concurrency note: single-flight operates within one process only.
        Across multiple processes, duplicate ``compute`` calls can still occur.
        """
        ns = _validate_namespace(namespace or self.namespace)

        # A bypass must be a true no-cache operation: do not generate an
        # embedding, touch the store, or join an ordinary request's inflight
        # computation.  Globally disabled caches use the same uncached path.
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

        cached = self.get(query, ns)
        if cached:
            return cached

        flight_key = f"{ns}:{query}"

        # Each inflight slot is a two-element list: [Event, result_or_exc|None].
        # Using a list (not just the Event) so the computing thread can store its
        # result in-place while waiters hold a reference to the same slot.
        # This avoids the race where result was popped from a separate dict before
        # the waiter could read it.
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

    def cached(self, namespace: str | None = None, bypass: bool = False):
        """Decorator for any ``fn(prompt: str) -> str`` (e.g. your LLM call).

        Usage::

            cache = SemanticCache(threshold=0.85)

            @cache.cached(namespace="gpt4:v1")
            def ask_llm(prompt: str) -> str:
                return openai.chat(prompt)

        The decorated function raises the original exception unchanged if the
        compute function fails.
        """
        import functools

        def wrap(fn: Callable[[str], str]) -> Callable[[str], str]:
            @functools.wraps(fn)
            def inner(prompt: str) -> str:
                return self.get_or_compute(
                    prompt, fn, namespace, bypass=bypass
                ).response

            return inner

        return wrap

    async def async_get_or_compute(
        self,
        query: str,
        compute: AsyncCompute,
        namespace: str | None = None,
        bypass: bool = False,
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

        cached = await asyncio.to_thread(self.get, query, ns)
        if cached:
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

    def async_cached(self, namespace: str | None = None, bypass: bool = False):
        """Decorator for ``async def fn(prompt: str) -> str`` functions."""
        import functools

        def wrap(fn: AsyncCompute) -> AsyncCompute:
            @functools.wraps(fn)
            async def inner(prompt: str) -> str:
                return (await self.async_get_or_compute(
                    prompt, fn, namespace, bypass=bypass
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
