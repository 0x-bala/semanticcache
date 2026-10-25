"""LangChain cache adapter for semantic-cache."""
from __future__ import annotations

import json
from typing import Any

from semantic_cache.cache import SemanticCache

try:
    from langchain_core.caches import BaseCache
    from langchain_core.outputs import Generation
except ImportError:
    # Duck-typed fallback if langchain_core is not installed
    class BaseCache:  # type: ignore[no-redef]
        pass

    class Generation:  # type: ignore[no-redef]
        def __init__(self, text: str, generation_info: dict | None = None) -> None:
            self.text = text
            self.generation_info = generation_info


class LangChainSemanticCache(BaseCache):
    """SemanticCache adapter for LangChain LLM calls."""

    def __init__(self, cache: SemanticCache | None = None, namespace: str = "langchain") -> None:
        if cache is None:
            from semantic_cache.cache import _get_default_cache

            cache = _get_default_cache()
        self.cache = cache
        self.namespace = namespace

    def _get_key(self, prompt: str, llm_string: str) -> str:
        return f"{llm_string}\n{prompt}"

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        """Look up based on prompt and llm_string."""
        key = self._get_key(prompt, llm_string)
        res = self.cache.get(key, namespace=self.namespace)
        if res is not None and res.hit:
            try:
                data = json.loads(res.response)
                if isinstance(data, list):
                    return [
                        Generation(text=item["text"], generation_info=item.get("generation_info"))
                        for item in data
                    ]
            except Exception:
                pass
            return [Generation(text=res.response)]
        return None

    def update(self, prompt: str, llm_string: str, return_val: list[Generation]) -> None:
        """Update cache based on prompt and llm_string."""
        key = self._get_key(prompt, llm_string)
        serialized = json.dumps(
            [
                {"text": gen.text, "generation_info": getattr(gen, "generation_info", None)}
                for gen in return_val
            ]
        )
        self.cache.set(key, serialized, namespace=self.namespace)

    def clear(self, **kwargs: Any) -> None:
        """Clear cache entries."""
        self.cache.clear(namespace=self.namespace)

    async def alookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        """Async lookup based on prompt and llm_string."""
        key = self._get_key(prompt, llm_string)
        res = await self.cache.aget(key, namespace=self.namespace)
        if res is not None and res.hit:
            try:
                data = json.loads(res.response)
                if isinstance(data, list):
                    return [
                        Generation(text=item["text"], generation_info=item.get("generation_info"))
                        for item in data
                    ]
            except Exception:
                pass
            return [Generation(text=res.response)]
        return None

    async def aupdate(self, prompt: str, llm_string: str, return_val: list[Generation]) -> None:
        """Async update cache based on prompt and llm_string."""
        key = self._get_key(prompt, llm_string)
        serialized = json.dumps(
            [
                {"text": gen.text, "generation_info": getattr(gen, "generation_info", None)}
                for gen in return_val
            ]
        )
        await self.cache.aset(key, serialized, namespace=self.namespace)

    async def aclear(self, **kwargs: Any) -> None:
        """Async clear cache entries."""
        self.cache.clear(namespace=self.namespace)
