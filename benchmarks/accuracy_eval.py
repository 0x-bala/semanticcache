"""Accuracy evaluation that drives the REAL ``SemanticCache.get()`` decision path.

The earlier Part 2 benchmark re-implemented the accept/verify rules by hand and
hard-coded "VERIFY counts as a hit above threshold + 0.05", which is not what the
library does. This module calls the shipped cache so that the numbers describe the
library's actual behaviour. It needs only numpy and ``semantic_cache`` (no torch).

Configurations per threshold
  off          guard disabled (pure vector similarity)
  default      guard on, library defaults (verify_always=True, no verifier):
               ACCEPT -> hit, VERIFY -> miss, REJECT -> miss
  legacy       guard on, verify_always=False, no verifier: VERIFY matches above
               threshold + guard_buffer are served unverified (the old behaviour)
  verifier     guard on, defaults, with the supplied verifier (optional)
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from semantic_cache import SemanticCache
from semantic_cache.guard import check_match
from semantic_cache.store import SQLiteStore


class LookupEmbedder:
    """Returns precomputed vectors for known texts (no model needed at run time)."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors
        self.dim = len(next(iter(vectors.values()))) if vectors else 0

    def embed(self, text: str) -> np.ndarray:
        vec = self._vectors.get(text)
        if vec is None:
            return np.zeros(self.dim, dtype=np.float32)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


def _metrics(true_hits: int, wrong_hits: int, dups: int, non_dups: int) -> dict[str, Any]:
    recall = true_hits / dups if dups else 0.0
    false_rate = wrong_hits / non_dups if non_dups else 0.0
    precision = true_hits / max(1, true_hits + wrong_hits)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    return {
        "true_duplicates_total": dups,
        "true_hits": true_hits,
        "recall": round(recall, 4),
        "non_duplicates_total": non_dups,
        "wrong_hits": wrong_hits,
        "false_hit_rate": round(false_rate, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def evaluate_pairs(
    pairs: list[dict[str, Any]],
    query_vecs: np.ndarray,
    corpus_questions: list[str],
    corpus_vecs: np.ndarray,
    thresholds: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
    verifier: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate labeled pairs through ``SemanticCache.get`` for every configuration."""
    dim = corpus_vecs.shape[1]
    index_of = {q: i for i, q in enumerate(corpus_questions[: len(corpus_vecs)])}
    vectors: dict[str, np.ndarray] = {}
    for p in pairs:
        vectors[p["cached_q"]] = corpus_vecs[index_of[p["cached_q"]]]
    for i, p in enumerate(pairs):
        vectors.setdefault(p["query"], query_vecs[i])
    embedder = LookupEmbedder(vectors)
    unique_cached = list(dict.fromkeys(p["cached_q"] for p in pairs))

    dups = sum(1 for p in pairs if p["is_duplicate"] == 1)
    non_dups = len(pairs) - dups

    verdict_counts = {"ACCEPT": 0, "VERIFY": 0, "REJECT": 0}
    for p in pairs:
        if p["is_duplicate"] == 1:
            verdict_counts[check_match(p["query"], p["cached_q"]).verdict.value] += 1

    configs: list[tuple[str, dict[str, Any]]] = [
        ("guard_off", {"guard": False}),
        ("guard_on", {"guard": True}),
        ("guard_on_legacy", {"guard": True, "verify_always": False}),
    ]
    if verifier is not None:
        configs.append(("guard_on_verifier", {"guard": True, "verifier": verifier}))

    settings: dict[str, Any] = {}
    worst: dict[str, list[dict[str, Any]]] = {}
    for name, kwargs in configs:
        for thresh in thresholds:
            store = SQLiteStore(path=":memory:", embedding_dim=dim, batch_writes=False)
            cache = SemanticCache(
                embedder=embedder, store=store, threshold=thresh, exact_match=False, **kwargs
            )
            for cq in unique_cached:
                cache.set(cq, f"Answer to {cq}")
            true_hits = wrong_hits = 0
            wrong: list[dict[str, Any]] = []
            for p in pairs:
                res = cache.get(p["query"])
                hit = res is not None and res.hit
                if not hit:
                    continue
                if p["is_duplicate"] == 1:
                    true_hits += 1
                else:
                    wrong_hits += 1
                    verdict = check_match(p["query"], res.matched_query).verdict.value
                    wrong.append(
                        {
                            "query": p["query"],
                            "matched_question": res.matched_query,
                            "score": round(float(res.score), 4),
                            "guard_verdict": verdict,
                            "ground_truth": 0,
                        }
                    )
            key = f"{name}_thresh_{int(round(thresh * 100))}"
            settings[key] = {
                "config": name,
                "threshold": thresh,
                "guard_enabled": kwargs["guard"],
                **_metrics(true_hits, wrong_hits, dups, non_dups),
            }
            if thresh == 0.85:
                wrong.sort(key=lambda w: w["score"], reverse=True)
                worst[name] = wrong[:20]
            cache.close() if hasattr(cache, "close") else store.close()
    return {
        "guard_duplicate_verdict_counts": verdict_counts,
        "settings": settings,
        "worst_20_wrong_hits_at_0.85": worst,
        "note": (
            "Decisions come from SemanticCache.get(). 'guard_on' uses library defaults "
            "(VERIFY without a verifier is a miss); 'guard_on_legacy' reproduces the old "
            "benchmark policy (verify_always=False)."
        ),
    }
