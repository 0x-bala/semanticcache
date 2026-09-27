"""Core semantic cache: look up answers by meaning, not exact text."""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Callable

from .embedders import Embedder, HashingEmbedder
from .store import SQLiteStore


@dataclass
class Stats:
    hits: int = 0
    misses: int = 0
    seconds_saved: float = 0.0  # estimated from the original call latency
    _latency: dict = field(default_factory=dict, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class CacheResult:
    response: str
    hit: bool
    score: float = 0.0
    matched_query: str | None = None


class SemanticCache:
    """
    threshold: minimum cosine similarity to count as a hit (0..1). Higher = safer.
    ttl:       seconds before entries expire (None = never).
    namespace: default partition. Use a different one per model/system prompt/user
               so unrelated requests never share answers.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: SQLiteStore | None = None,
        threshold: float = 0.85,
        ttl: float | None = None,
        namespace: str = "default",
    ):
        self.embedder = embedder or HashingEmbedder()
        self.store = store or SQLiteStore()
        self.threshold = threshold
        self.ttl = ttl
        self.namespace = namespace
        self.stats = Stats()

    def get(self, query: str, namespace: str | None = None) -> CacheResult | None:
        ns = namespace or self.namespace
        entry = self.store.search(ns, self.embedder.embed(query))
        if entry and entry.score >= self.threshold:
            self.store.record_hit(entry.id)
            self.stats.hits += 1
            self.stats.seconds_saved += self.stats._latency.get(entry.id, 0.0)
            return CacheResult(entry.response, True, entry.score, entry.query)
        self.stats.misses += 1
        return None

    def set(self, query: str, response: str, namespace: str | None = None, latency: float = 0.0) -> None:
        ns = namespace or self.namespace
        entry_id = self.store.add(ns, query, self.embedder.embed(query), response, self.ttl)
        self.stats._latency[entry_id] = latency

    def get_or_compute(self, query: str, compute: Callable[[str], str], namespace: str | None = None) -> CacheResult:
        cached = self.get(query, namespace)
        if cached:
            return cached
        start = time.perf_counter()
        response = compute(query)
        self.set(query, response, namespace, latency=time.perf_counter() - start)
        return CacheResult(response, False)

    def cached(self, namespace: str | None = None):
        """Decorator for any `fn(prompt: str) -> str` (e.g. your LLM call)."""

        def wrap(fn):
            @functools.wraps(fn)
            def inner(prompt: str) -> str:
                return self.get_or_compute(prompt, fn, namespace).response

            return inner

        return wrap

    def clear(self, namespace: str | None = None) -> None:
        self.store.clear(namespace)
