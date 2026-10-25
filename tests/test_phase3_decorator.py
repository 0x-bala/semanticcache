"""Tests for standalone @cached decorator."""
from __future__ import annotations

import asyncio

import pytest

from semantic_cache import HashingEmbedder, SemanticCache, SQLiteStore, cached


def test_cached_decorator_sync_function():
    store = SQLiteStore(path=":memory:", embedding_dim=16)
    cache = SemanticCache(embedder=HashingEmbedder(dim=16), store=store, threshold=0.85)
    calls = 0

    @cached(cache=cache)
    def compute_answer(query: str) -> str:
        nonlocal calls
        calls += 1
        return f"Computed: {query}"

    expected = "Computed: What is life?"
    assert compute_answer("What is life?") == expected
    assert calls == 1

    # Exact hit
    assert compute_answer("What is life?") == expected
    assert calls == 1

    # Paraphrased semantic hit
    assert compute_answer("What is life please?") == expected
    assert calls == 1

    # Bypass flag ignores cache and computes fresh
    assert compute_answer("What is life?", bypass=True) == expected
    assert calls == 2

    cache.close()


@pytest.mark.anyio
async def test_cached_decorator_async_function():
    store = SQLiteStore(path=":memory:", embedding_dim=16)
    cache = SemanticCache(embedder=HashingEmbedder(dim=16), store=store, threshold=0.85)
    calls = 0

    @cached(cache=cache, namespace="async_ns")
    async def async_compute_answer(prompt: str) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return f"Async computed: {prompt}"

    res1 = await async_compute_answer(prompt="Tell me a joke")
    assert res1 == "Async computed: Tell me a joke"
    assert calls == 1

    res2 = await async_compute_answer(prompt="Tell me a joke")
    assert res2 == "Async computed: Tell me a joke"
    assert calls == 1

    res_para = await async_compute_answer(prompt="Tell me a joke please")
    assert res_para == "Async computed: Tell me a joke"
    assert calls == 1

    cache.close()


def test_cached_decorator_custom_key_fn():
    store = SQLiteStore(path=":memory:", embedding_dim=16)
    cache = SemanticCache(embedder=HashingEmbedder(dim=16), store=store)
    calls = 0

    @cached(cache=cache, key_fn=lambda doc, mode="fast": f"{mode}:{doc['text']}")
    def process_doc(doc: dict, mode: str = "fast") -> str:
        nonlocal calls
        calls += 1
        return f"Processed {doc['text']} ({mode})"

    doc1 = {"text": "hello world"}
    res1 = process_doc(doc1, mode="fast")
    assert res1 == "Processed hello world (fast)"
    assert calls == 1

    res2 = process_doc(doc1, mode="fast")
    assert res2 == "Processed hello world (fast)"
    assert calls == 1

    cache.close()
