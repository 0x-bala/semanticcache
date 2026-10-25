"""Tests for LFU, FIFO, and LRU cache eviction policies."""
from __future__ import annotations

import time

import numpy as np
import pytest

from semantic_cache.store import SQLiteStore


def test_lfu_eviction_removes_least_frequently_used():
    dim = 4
    store = SQLiteStore(path=":memory:", embedding_dim=dim, max_entries=3, eviction_policy="lfu")
    v = np.zeros(dim, dtype=np.float32)

    # Insert 3 entries
    id1 = store.add("ns", "q1", v, "r1", ttl=None)
    store.add("ns", "q2", v, "r2", ttl=None)
    id3 = store.add("ns", "q3", v, "r3", ttl=None)

    # Record hits: q1 gets 5 hits, q3 gets 2 hits, q2 gets 0 hits
    for _ in range(5):
        store.record_hit(id1)
    for _ in range(2):
        store.record_hit(id3)

    # Insert 4th entry to trigger LFU eviction down to 90% capacity (< 3, i.e. 2 entries remain)
    store.add("ns", "q4", v, "r4", ttl=None)

    # q2 (0 hits) should be evicted
    assert store.find_exact("ns", query_hash=store._exact_hash("q2")) is None
    # q1 (5 hits, highest frequency) must still be present
    assert store.find_exact("ns", query_hash=store._exact_hash("q1")) is not None

    store.close()


def test_fifo_eviction_removes_oldest_regardless_of_hits():
    dim = 4
    store = SQLiteStore(path=":memory:", embedding_dim=dim, max_entries=3, eviction_policy="fifo")
    v = np.zeros(dim, dtype=np.float32)

    id1 = store.add("ns", "q1", v, "r1", ttl=None)
    time.sleep(0.01)
    store.add("ns", "q2", v, "r2", ttl=None)
    time.sleep(0.01)
    store.add("ns", "q3", v, "r3", ttl=None)

    # Even though q1 is hit 10 times, FIFO strictly evicts oldest created entry first
    for _ in range(10):
        store.record_hit(id1)

    # Adding q4 triggers FIFO eviction
    store.add("ns", "q4", v, "r4", ttl=None)

    # Oldest entry q1 must be evicted
    assert store.find_exact("ns", query_hash=store._exact_hash("q1")) is None
    # Newer entries q3 and q4 must remain
    assert store.find_exact("ns", query_hash=store._exact_hash("q3")) is not None
    assert store.find_exact("ns", query_hash=store._exact_hash("q4")) is not None

    store.close()


def test_invalid_eviction_policy_raises():
    with pytest.raises(ValueError, match="Invalid eviction_policy"):
        SQLiteStore(eviction_policy="random_policy")
