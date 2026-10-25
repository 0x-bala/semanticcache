"""Tests for async API (aget, aset, aask) and streaming-response caching."""
from __future__ import annotations

import asyncio

import pytest

from semantic_cache import HashingEmbedder, SemanticCache, SQLiteStore


@pytest.mark.anyio
async def test_aget_and_aset():
    cache = SemanticCache(
        embedder=HashingEmbedder(dim=16),
        store=SQLiteStore(path=":memory:", embedding_dim=16),
    )
    await cache.aset("What is async Python?", "Async Python uses coroutines.")
    res = await cache.aget("What is async Python?")
    assert res is not None
    assert res.hit is True
    assert res.response == "Async Python uses coroutines."

    # Semantic paraphrase hit
    res_para = await cache.aget("What is async Python please?")
    assert res_para is not None
    assert res_para.hit is True

    # Miss
    miss = await cache.aget("Something completely different")
    assert miss is None
    cache.close()


@pytest.mark.anyio
async def test_aask_with_sync_and_async_callable():
    cache = SemanticCache(
        embedder=HashingEmbedder(dim=16),
        store=SQLiteStore(path=":memory:", embedding_dim=16),
    )
    sync_calls = 0

    def sync_llm(prompt: str) -> str:
        nonlocal sync_calls
        sync_calls += 1
        return f"Sync answer to {prompt}"

    # First call computes
    ans1 = await cache.aask("Question 1", sync_llm)
    assert ans1 == "Sync answer to Question 1"
    assert sync_calls == 1

    # Second call hits cache
    ans2 = await cache.aask("Question 1", sync_llm)
    assert ans2 == "Sync answer to Question 1"
    assert sync_calls == 1

    async_calls = 0

    async def async_llm(prompt: str) -> str:
        nonlocal async_calls
        async_calls += 1
        await asyncio.sleep(0.01)
        return f"Async answer to {prompt}"

    ans3 = await cache.aask("Question 2", async_llm)
    assert ans3 == "Async answer to Question 2"
    assert async_calls == 1

    ans4 = await cache.aask("Question 2", async_llm)
    assert ans4 == "Async answer to Question 2"
    assert async_calls == 1

    cache.close()


def test_sync_streaming_response_caching():
    cache = SemanticCache(
        embedder=HashingEmbedder(dim=16),
        store=SQLiteStore(path=":memory:", embedding_dim=16),
    )
    called = 0

    def stream_gen(prompt: str):
        nonlocal called
        called += 1
        yield "Hello "
        yield "world "
        yield "from "
        yield "stream!"

    # First call streams from generator
    chunks1 = list(cache.get_or_compute_stream("Stream prompt", stream_gen))
    assert "".join(chunks1) == "Hello world from stream!"
    assert called == 1

    # Second call returns full cached response as single stream chunk without re-calling generator
    chunks2 = list(cache.get_or_compute_stream("Stream prompt", stream_gen))
    assert "".join(chunks2) == "Hello world from stream!"
    assert called == 1

    # Paraphrased semantic hit also receives cached stream
    chunks3 = list(cache.get_or_compute_stream("Stream prompt please", stream_gen))
    assert "".join(chunks3) == "Hello world from stream!"
    assert called == 1

    cache.close()


@pytest.mark.anyio
async def test_async_streaming_response_caching():
    cache = SemanticCache(
        embedder=HashingEmbedder(dim=16),
        store=SQLiteStore(path=":memory:", embedding_dim=16),
    )
    called = 0

    async def async_stream_gen(prompt: str):
        nonlocal called
        called += 1
        for part in ["Chunk 1 ", "Chunk 2 ", "Chunk 3"]:
            await asyncio.sleep(0.005)
            yield part

    # First call streams from generator
    result_chunks1 = []
    async for chunk in cache.async_get_or_compute_stream("Async stream prompt", async_stream_gen):
        result_chunks1.append(chunk)

    assert "".join(result_chunks1) == "Chunk 1 Chunk 2 Chunk 3"
    assert called == 1

    # Second call receives cached result without calling generator
    result_chunks2 = []
    async for chunk in cache.async_get_or_compute_stream("Async stream prompt", async_stream_gen):
        result_chunks2.append(chunk)

    assert "".join(result_chunks2) == "Chunk 1 Chunk 2 Chunk 3"
    assert called == 1

    cache.close()
