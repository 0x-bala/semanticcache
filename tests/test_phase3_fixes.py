"""Regression tests for bugs found when auditing the Phase 1-3 commits."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import numpy as np
import pytest

from semantic_cache import SemanticCache, SQLiteStore, cached


def test_cached_does_not_share_answers_across_other_arguments():
    calls: list[str] = []

    @cached(cache=SemanticCache(threshold=0.9))
    def ask(prompt, model="small"):
        calls.append(model)
        return f"answer from {model}"

    assert ask("capital of France?", model="small") == "answer from small"
    assert ask("capital of France?", model="large") == "answer from large"
    assert ask("capital of France?", model="small") == "answer from small"  # real hit
    assert calls == ["small", "large"]


def test_cached_on_method_keys_on_prompt_not_self():
    class Client:
        @cached(cache=SemanticCache(threshold=0.9))
        def ask(self, prompt):
            return "answer to: " + prompt

    c = Client()
    assert c.ask("what is the capital of France") == "answer to: what is the capital of France"
    assert c.ask("how do I bake bread") == "answer to: how do I bake bread"


def test_async_cached_honours_ttl():
    cache = SemanticCache(threshold=0.9)

    @cached(cache=cache, ttl=0.2)
    async def aq(prompt):
        return "x"

    async def run():
        await aq("hello ttl")
        before = cache.get("hello ttl")
        await asyncio.sleep(0.4)
        return before, cache.get("hello ttl")

    before, after = asyncio.run(run())
    assert before is not None
    assert after is None


@pytest.mark.parametrize("policy", ["lru", "lfu", "fifo"])
def test_new_entry_survives_eviction_even_when_others_have_hits(policy):
    rng = np.random.default_rng(0)

    def vec():
        x = rng.normal(size=8).astype("float32")
        return x / np.linalg.norm(x)

    store = SQLiteStore(max_entries=10, embedding_dim=8, eviction_policy=policy)
    ids = [store.add("n", f"q{i}", vec(), "r", None) for i in range(10)]
    for eid in ids:
        store.record_hit(eid)
    store.add("n", "brandnew", vec(), "r", None)
    assert store.find_exact("n", SQLiteStore._exact_hash("brandnew")) is not None
    store.close()


def test_pending_writes_survive_exit_without_close(tmp_path):
    path = str(tmp_path / "c.db")
    code = (
        "from semantic_cache import SemanticCache, SQLiteStore\n"
        f"c = SemanticCache(store=SQLiteStore(path={path!r}))\n"
        "c.set('normal exit question', 'ans')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    with SQLiteStore(path=path) as store:
        assert store.size() == 1


def test_hnswlib_backend_is_used_and_sees_late_entries():
    pytest.importorskip("hnswlib")
    rng = np.random.default_rng(1)

    def unit(n, d):
        x = rng.normal(size=(n, d)).astype("float32")
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    store = SQLiteStore(embedding_dim=32, ann_backend="hnswlib", ann_threshold=50)
    for i, v in enumerate(unit(200, 32)):
        store.add("n", f"q{i}", v, f"r{i}", None)
    late = unit(1, 32)[0]
    store.add("n", "late", late, "late-resp", None)
    index = store._index["n"]
    assert hasattr(index.ann_index, "knn_query")  # a real hnswlib index, not a silent fallback
    hits = store.search_candidates("n", late, top_k=1)
    assert hits and hits[0].response == "late-resp"
    store.close()


def test_batched_hit_counts_are_not_collapsed(tmp_path):
    store = SQLiteStore(path=str(tmp_path / "h.db"), embedding_dim=4)
    eid = store.add("n", "q", np.ones(4, dtype="float32") / 2, "r", None)
    for _ in range(5):
        store.record_hit(eid)
    store.flush()
    assert store.db.execute("SELECT hits FROM entries WHERE id=?", (eid,)).fetchone()[0] == 5
    store.close()
