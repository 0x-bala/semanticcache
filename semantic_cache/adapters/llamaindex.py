"""LlamaIndex cache adapter for semantic-cache."""
from __future__ import annotations

import json
from typing import Any

from semantic_cache.cache import SemanticCache

try:
    from llama_index.core.storage.kvstore.types import BaseKVStore
except ImportError:
    class BaseKVStore:  # type: ignore[no-redef]
        pass


class LlamaIndexSemanticCache(BaseKVStore):
    """SemanticCache adapter for LlamaIndex KVStore and prompt response caching."""

    def __init__(self, cache: SemanticCache | None = None, namespace: str = "llamaindex") -> None:
        if cache is None:
            from semantic_cache.cache import _get_default_cache

            cache = _get_default_cache()
        self.cache = cache
        self.namespace = namespace

    def put(self, key: str, val: dict[str, Any], collection: str | None = None) -> None:
        """Store a dictionary value."""
        ns = f"{self.namespace}::{collection}" if collection else self.namespace
        self.cache.set(key, json.dumps(val), namespace=ns)

    def get(self, key: str, collection: str | None = None) -> dict[str, Any] | None:
        """Retrieve a dictionary value."""
        ns = f"{self.namespace}::{collection}" if collection else self.namespace
        res = self.cache.get(key, namespace=ns)
        if res is not None and res.hit:
            try:
                return json.loads(res.response)
            except Exception:
                return {"response": res.response}
        return None

    def delete(self, key: str, collection: str | None = None) -> bool:
        """Delete an entry."""
        ns = f"{self.namespace}::{collection}" if collection else self.namespace
        return self.cache.delete(key, namespace=ns) > 0

    async def aput(self, key: str, val: dict[str, Any], collection: str | None = None) -> None:
        """Async store a dictionary value."""
        ns = f"{self.namespace}::{collection}" if collection else self.namespace
        await self.cache.aset(key, json.dumps(val), namespace=ns)

    async def aget(self, key: str, collection: str | None = None) -> dict[str, Any] | None:
        """Async retrieve a dictionary value."""
        ns = f"{self.namespace}::{collection}" if collection else self.namespace
        res = await self.cache.aget(key, namespace=ns)
        if res is not None and res.hit:
            try:
                return json.loads(res.response)
            except Exception:
                return {"response": res.response}
        return None

    async def adelete(self, key: str, collection: str | None = None) -> bool:
        """Async delete an entry."""
        ns = f"{self.namespace}::{collection}" if collection else self.namespace
        return self.cache.delete(key, namespace=ns) > 0
