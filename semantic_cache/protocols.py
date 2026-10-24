"""Public protocols for implementing custom semantic-cache backends."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .store import Entry


@runtime_checkable
class CacheStore(Protocol):
    """Storage contract used by :class:`SemanticCache`.

    Implement this protocol to add a Redis, Postgres, or vector-database
    backend without changing the cache orchestration layer.
    """

    def add(
        self,
        namespace: str,
        query: str,
        embedding: np.ndarray,
        response: str,
        ttl: float | None,
    ) -> int: ...

    def search(self, namespace: str, embedding: np.ndarray) -> Entry | None: ...

    def record_hit(self, entry_id: int) -> None: ...

    def clear(self, namespace: str | None = None) -> None: ...

    def delete_by_query(self, namespace: str, query: str) -> int: ...

    def invalidate_expired(self) -> int: ...

    def namespace_stats(self, namespace: str) -> dict: ...
