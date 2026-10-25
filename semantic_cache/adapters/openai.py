"""OpenAI client wrapper for semantic caching."""
from __future__ import annotations

import functools
from collections.abc import AsyncIterator, Iterator
from typing import Any

from semantic_cache.cache import SemanticCache


def _extract_query_from_chat_params(messages: list[dict[str, Any]] | list[Any]) -> str:
    """Extract a representative query string from chat completion messages."""
    if not messages:
        return ""
    # Use full message history as JSON if multiple messages, or last user message if single
    serialized = []
    for m in messages:
        if isinstance(m, dict):
            serialized.append(f"{m.get('role', '')}: {m.get('content', '')}")
        else:
            role = getattr(m, "role", "")
            content = getattr(m, "content", "")
            serialized.append(f"{role}: {content}")
    return "\n".join(serialized)


class _CachedChoice:
    def __init__(self, content: str, index: int = 0) -> None:
        self.index = index
        self.message = _CachedMessage(content)
        self.finish_reason = "stop"


class _CachedMessage:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content


class _CachedChatCompletion:
    def __init__(self, content: str, model: str = "cached-model") -> None:
        self.id = "cached-completion"
        self.object = "chat.completion"
        self.created = 0
        self.model = model
        self.choices = [_CachedChoice(content)]


class _CachedChunkChoice:
    def __init__(self, delta_content: str, index: int = 0) -> None:
        self.index = index
        self.delta = _CachedMessage(delta_content)
        self.finish_reason = None


class _CachedChatCompletionChunk:
    def __init__(self, delta_content: str, model: str = "cached-model") -> None:
        self.id = "cached-chunk"
        self.object = "chat.completion.chunk"
        self.created = 0
        self.model = model
        self.choices = [_CachedChunkChoice(delta_content)]


def wrap_openai(client: Any, cache: SemanticCache | None = None) -> Any:
    """Wrap an OpenAI or AsyncOpenAI client instance to automatically cache completions."""
    if cache is None:
        from semantic_cache.cache import _get_default_cache

        cache = _get_default_cache()

    original_create = client.chat.completions.create

    import inspect

    if inspect.iscoroutinefunction(original_create):
        @functools.wraps(original_create)
        async def async_cached_create(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages") or (args[0] if args else [])
            stream = bool(kwargs.get("stream", False))
            query = _extract_query_from_chat_params(messages)
            model = kwargs.get("model", "unknown-model")
            namespace = f"openai::{model}"

            if not query:
                return await original_create(*args, **kwargs)

            if stream:
                cached_res = await cache.aget(query, namespace=namespace)
                if cached_res is not None and cached_res.hit:
                    async def _yield_cached() -> AsyncIterator[Any]:
                        yield _CachedChatCompletionChunk(cached_res.response, model=model)
                    return _yield_cached()

                raw_stream = await original_create(*args, **kwargs)

                async def _intercept_stream() -> AsyncIterator[Any]:
                    accumulated: list[str] = []
                    async for chunk in raw_stream:
                        yield chunk
                        choices = getattr(chunk, "choices", [])
                        if choices:
                            delta = getattr(choices[0], "delta", None)
                            c = getattr(delta, "content", None) if delta else None
                            if c:
                                accumulated.append(c)
                    full_text = "".join(accumulated)
                    if full_text:
                        await cache.aset(query, full_text, namespace=namespace)

                return _intercept_stream()

            cached_res = await cache.aget(query, namespace=namespace)
            if cached_res is not None and cached_res.hit:
                return _CachedChatCompletion(cached_res.response, model=model)

            real_completion = await original_create(*args, **kwargs)
            try:
                content = real_completion.choices[0].message.content
                if content:
                    await cache.aset(query, content, namespace=namespace)
            except Exception:
                pass
            return real_completion

        client.chat.completions.create = async_cached_create
    else:
        @functools.wraps(original_create)
        def sync_cached_create(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages") or (args[0] if args else [])
            stream = bool(kwargs.get("stream", False))
            query = _extract_query_from_chat_params(messages)
            model = kwargs.get("model", "unknown-model")
            namespace = f"openai::{model}"

            if not query:
                return original_create(*args, **kwargs)

            if stream:
                cached_res = cache.get(query, namespace=namespace)
                if cached_res is not None and cached_res.hit:
                    def _yield_cached() -> Iterator[Any]:
                        yield _CachedChatCompletionChunk(cached_res.response, model=model)
                    return _yield_cached()

                raw_stream = original_create(*args, **kwargs)

                def _intercept_stream() -> Iterator[Any]:
                    accumulated: list[str] = []
                    for chunk in raw_stream:
                        yield chunk
                        choices = getattr(chunk, "choices", [])
                        if choices:
                            delta = getattr(choices[0], "delta", None)
                            c = getattr(delta, "content", None) if delta else None
                            if c:
                                accumulated.append(c)
                    full_text = "".join(accumulated)
                    if full_text:
                        cache.set(query, full_text, namespace=namespace)

                return _intercept_stream()

            cached_res = cache.get(query, namespace=namespace)
            if cached_res is not None and cached_res.hit:
                return _CachedChatCompletion(cached_res.response, model=model)

            real_completion = original_create(*args, **kwargs)
            try:
                content = real_completion.choices[0].message.content
                if content:
                    cache.set(query, content, namespace=namespace)
            except Exception:
                pass
            return real_completion

        client.chat.completions.create = sync_cached_create

    return client
