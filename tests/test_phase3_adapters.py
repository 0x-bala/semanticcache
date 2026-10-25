"""Tests for OpenAI wrapper, LangChain adapter, and LlamaIndex adapter."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantic_cache import HashingEmbedder, SemanticCache, SQLiteStore
from semantic_cache.adapters import LangChainSemanticCache, LlamaIndexSemanticCache, wrap_openai


def test_wrap_openai_sync():
    store = SQLiteStore(path=":memory:", embedding_dim=16)
    cache = SemanticCache(embedder=HashingEmbedder(dim=16), store=store)

    calls = 0
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock()]
    fake_completion.choices[0].message.content = "OpenAI answer about quantum physics"

    fake_client = MagicMock()

    def fake_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_completion

    fake_client.chat.completions.create = fake_create

    wrapped = wrap_openai(fake_client, cache=cache)

    messages = [{"role": "user", "content": "Explain quantum physics"}]
    res1 = wrapped.chat.completions.create(model="gpt-4o", messages=messages)
    assert res1.choices[0].message.content == "OpenAI answer about quantum physics"
    assert calls == 1

    # Second call served from semantic cache!
    res2 = wrapped.chat.completions.create(model="gpt-4o", messages=messages)
    assert res2.choices[0].message.content == "OpenAI answer about quantum physics"
    assert calls == 1

    cache.close()


def test_wrap_openai_sync_stream():
    store = SQLiteStore(path=":memory:", embedding_dim=16)
    cache = SemanticCache(embedder=HashingEmbedder(dim=16), store=store)

    calls = 0
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Streaming "
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "answer"

    fake_client = MagicMock()

    def fake_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [chunk1, chunk2]

    fake_client.chat.completions.create = fake_create

    wrapped = wrap_openai(fake_client, cache=cache)
    messages = [{"role": "user", "content": "Tell me a stream story"}]

    stream1 = wrapped.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
    pieces1 = [c.choices[0].delta.content for c in stream1]
    assert "".join(pieces1) == "Streaming answer"
    assert calls == 1

    # Second streaming call served from semantic cache without calling provider
    stream2 = wrapped.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
    pieces2 = [c.choices[0].delta.content for c in stream2]
    assert "".join(pieces2) == "Streaming answer"
    assert calls == 1

    cache.close()


@pytest.mark.anyio
async def test_wrap_openai_async():
    store = SQLiteStore(path=":memory:", embedding_dim=16)
    cache = SemanticCache(embedder=HashingEmbedder(dim=16), store=store)

    calls = 0
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock()]
    fake_completion.choices[0].message.content = "Async answer"

    async def fake_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_completion

    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create

    wrapped = wrap_openai(fake_client, cache=cache)
    messages = [{"role": "user", "content": "Async prompt"}]

    res1 = await wrapped.chat.completions.create(model="gpt-4o", messages=messages)
    assert res1.choices[0].message.content == "Async answer"
    assert calls == 1

    res2 = await wrapped.chat.completions.create(model="gpt-4o", messages=messages)
    assert res2.choices[0].message.content == "Async answer"
    assert calls == 1

    cache.close()


def test_langchain_adapter_sync_and_async():
    from semantic_cache.adapters.langchain import Generation

    store = SQLiteStore(path=":memory:", embedding_dim=16)
    cache = SemanticCache(embedder=HashingEmbedder(dim=16), store=store)
    lc_cache = LangChainSemanticCache(cache=cache)

    prompt = "What is the capital of France?"
    llm_str = "gpt-4o-temp-0"

    assert lc_cache.lookup(prompt, llm_str) is None

    gen = [Generation(text="Paris is the capital of France.")]
    lc_cache.update(prompt, llm_str, gen)

    cached_gen = lc_cache.lookup(prompt, llm_str)
    assert cached_gen is not None
    assert cached_gen[0].text == "Paris is the capital of France."

    # Clear
    lc_cache.clear()
    assert lc_cache.lookup(prompt, llm_str) is None

    cache.close()


def test_llamaindex_adapter_sync_and_async():
    store = SQLiteStore(path=":memory:", embedding_dim=16)
    cache = SemanticCache(embedder=HashingEmbedder(dim=16), store=store)
    llama_cache = LlamaIndexSemanticCache(cache=cache)

    key = "query-prompt-1"
    val = {"answer": "LlamaIndex semantic cache answer", "score": 0.95}

    assert llama_cache.get(key) is None

    llama_cache.put(key, val)
    cached_val = llama_cache.get(key)
    assert cached_val is not None
    assert cached_val["answer"] == "LlamaIndex semantic cache answer"

    # Delete
    deleted = llama_cache.delete(key)
    assert deleted is True
    assert llama_cache.get(key) is None

    cache.close()
