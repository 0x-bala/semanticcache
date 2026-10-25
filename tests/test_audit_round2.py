"""Regression tests for the guard must-verify flag and lazy ANN eviction."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_cache import SemanticCache
from semantic_cache.guard import GuardVerdict, check_match
from semantic_cache.store import VectorIndex


class _ConstantEmbedder:
    """Every text gets the same vector, so every pair scores 1.0 similarity."""

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        return v

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


SWAP_PAIRS = [
    (
        "Is it possible for an Indian girl to be in a relationship with a Chinese boy?",
        "Is it possible for a Chinese girl to be in a relationship with an Indian boy?",
    ),
    ("What is the best way to learn French?", "What is the best way to learn Spanish?"),
]


@pytest.mark.parametrize("cached,new", SWAP_PAIRS)
def test_swaps_and_substitutions_are_flagged_must_verify(cached, new):
    result = check_match(new, cached)
    assert result.verdict == GuardVerdict.VERIFY
    assert result.must_verify is True


@pytest.mark.parametrize("cached,new", SWAP_PAIRS)
def test_high_score_swap_is_not_served_without_verification(cached, new):
    # verify_always=False used to let VERIFY matches above threshold+buffer through.
    cache = SemanticCache(embedder=_ConstantEmbedder(), threshold=0.85, verify_always=False)
    cache.set(cached, "ANSWER")
    assert cache.get(new) is None


@pytest.mark.parametrize("cached,new", SWAP_PAIRS)
def test_must_verify_matches_still_hit_when_verifier_accepts(cached, new):
    cache = SemanticCache(
        embedder=_ConstantEmbedder(),
        threshold=0.85,
        verify_always=False,
        verifier=lambda a, b: True,
    )
    cache.set(cached, "ANSWER")
    assert cache.get(new) is not None


def test_ordinary_verify_still_skips_verifier_above_buffer():
    assert check_match("tell me a joke about cats", "tell me a joke").must_verify is False


def _unit(rng, n, d):
    x = rng.normal(size=(n, d)).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize("backend", ["faiss", "hnswlib"])
def test_ann_eviction_filters_without_rebuilding_until_threshold(backend):
    pytest.importorskip("faiss" if backend == "faiss" else "hnswlib")
    rng = np.random.default_rng(3)
    d = 32
    vecs = _unit(rng, 200, d)
    index = VectorIndex(list(range(200)), vecs, d, ann_threshold=50, ann_backend=backend)
    assert index.ann_index is not None
    first_graph = index.ann_index

    index.remove_ids(set(range(30)))  # 15% < 25%: graph is kept, ids are filtered
    assert index.ann_index is first_graph
    for evicted in (0, 7, 29):
        ids, _ = index.search(vecs[evicted], top_k=3)
        assert evicted not in ids
    ids, _ = index.search(vecs[100], top_k=1)
    assert ids == [100]

    index.remove_ids(set(range(30, 80)))  # now > 25% evicted: graph is rebuilt
    assert index.ann_index is not first_graph
    assert index.ann_removed == set()
    assert index.ann_ids == index.ids
    ids, _ = index.search(vecs[150], top_k=1)
    assert ids == [150]
