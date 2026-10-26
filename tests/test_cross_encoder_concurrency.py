"""Deterministic concurrency and thread-safety regression tests for CrossEncoderVerifier.

Verifies:
- Concurrent requests for different query pairs never receive each other's scores.
- Inverted inference completion order preserves pair-score correspondence.
- Repeated requests for the same pair safely reuse cached score without duplicate inference.
- Concurrent calls through score() and __call__() cannot mix pair and score state.
- Model exceptions and malformed outputs do not poison the score cache and can be retried.
- Invalid and Boolean predictions remain rejected under concurrency.
- Concurrent SemanticCache.get() lookups maintain strict decision isolation.
- Bounded cache eviction remains strictly bounded under concurrent inserts.

Uses deterministic synchronization primitives (threading.Barrier, threading.Event).
No external APIs or network calls.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from semantic_cache.cache import SemanticCache
from semantic_cache.cross_encoder import CrossEncoderVerifier
from semantic_cache.store import SQLiteStore


def _make_cev(
    model=None, threshold: float = 0.85, max_cache_size: int = 1000
) -> CrossEncoderVerifier:
    """Helper to instantiate CrossEncoderVerifier with a mock model."""
    cev = object.__new__(CrossEncoderVerifier)
    cev.threshold = float(threshold)
    cev.max_cache_size = max_cache_size
    cev._cache = OrderedDict()
    cev._cache_lock = threading.Lock()
    cev._inflight = {}
    cev._last_pair = None
    cev._last_score = None
    if model is not None:
        cev.model = model
    return cev


DIM = 4


class MockEmbedder:
    dim = DIM

    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def embed(self, text: str) -> np.ndarray:
        return self.mapping.get(text, np.zeros(DIM, dtype=np.float32))


def _make_cache(embedder, verifier, top_k: int = 5) -> SemanticCache:
    store = SQLiteStore(path=":memory:", embedding_dim=DIM)
    return SemanticCache(
        embedder=embedder,
        store=store,
        threshold=0.85,
        exact_match=False,
        verifier=verifier,
        top_k=top_k,
    )


class TestCrossEncoderThreadSafety:
    """Thread safety and memoization race prevention in CrossEncoderVerifier."""

    def test_concurrent_different_pairs_never_receive_each_others_scores(self):
        """Concurrent requests for distinct pairs must strictly receive their own scores."""
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        pair_scores = {
            (f"q_{i}", f"cand_{i}"): round(0.50 + i * 0.04, 3) for i in range(num_threads)
        }

        class ConcurrentPredictModel:
            def predict(self, pairs):
                barrier.wait(timeout=5.0)
                pair = tuple(pairs[0])
                score = pair_scores[pair]
                return [score]

        cev = _make_cev(model=ConcurrentPredictModel(), threshold=0.85)
        results: dict[tuple[str, str], float | None] = {}
        errors: list[Exception] = []

        def worker(pair):
            try:
                res = cev.score(pair[0], pair[1])
                results[pair] = res
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(pair,)) for pair in pair_scores
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Workers encountered exceptions: {errors}"
        assert len(results) == num_threads

        for pair, expected in pair_scores.items():
            actual = results[pair]
            assert actual == pytest.approx(expected, abs=1e-4), (
                f"Pair {pair} received mismatched score {actual} instead of {expected}"
            )

    def test_inference_completion_order_inversion_preserves_pair_scores(self):
        """Even when inference completion order is inverted, pairs match their own scores."""
        pair_slow = ("query_slow", "cand_slow")
        pair_fast = ("query_fast", "cand_fast")

        slow_started = threading.Event()
        fast_can_proceed = threading.Event()
        fast_finished = threading.Event()

        class InvertedOrderModel:
            def predict(self, pairs):
                pair = tuple(pairs[0])
                if pair == pair_slow:
                    slow_started.set()
                    # Wait until the fast pair finishes before completing
                    fast_finished.wait(timeout=5.0)
                    return [0.95]
                elif pair == pair_fast:
                    # Fast pair waits for slow to start, then completes first
                    fast_can_proceed.wait(timeout=5.0)
                    res = [0.25]
                    fast_finished.set()
                    return res
                return [0.0]

        cev = _make_cev(model=InvertedOrderModel(), threshold=0.85)
        results: dict[tuple[str, str], float | None] = {}

        def slow_worker():
            results[pair_slow] = cev.score(*pair_slow)

        def fast_worker():
            slow_started.wait(timeout=5.0)
            fast_can_proceed.set()
            results[pair_fast] = cev.score(*pair_fast)

        t_slow = threading.Thread(target=slow_worker)
        t_fast = threading.Thread(target=fast_worker)

        t_slow.start()
        t_fast.start()

        t_slow.join(timeout=5.0)
        t_fast.join(timeout=5.0)

        assert results[pair_slow] == pytest.approx(0.95)
        assert results[pair_fast] == pytest.approx(0.25)

    def test_repeated_requests_for_same_pair_reuse_cached_score(self):
        """Concurrent requests for the exact same pair execute model inference only once."""
        num_threads = 8
        barrier = threading.Barrier(num_threads)

        class SingleInferenceCountingModel:
            def __init__(self):
                self.calls = 0
                self._lock = threading.Lock()

            def predict(self, pairs):
                with self._lock:
                    self.calls += 1
                return [0.93]

        model = SingleInferenceCountingModel()
        cev = _make_cev(model=model, threshold=0.85)
        results: list[float | None] = []

        def worker():
            barrier.wait(timeout=5.0)
            score = cev.score("same_query", "same_candidate")
            results.append(score)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(executor.map(lambda _: worker(), range(num_threads)))

        assert len(results) == num_threads
        assert all(r == pytest.approx(0.93) for r in results)
        assert model.calls == 1, f"Expected 1 model inference call, got {model.calls}"

    def test_concurrent_score_and_call_cannot_mix_state(self):
        """score() and __call__() running concurrently on different pairs cannot mix states."""
        pair_high = ("q_high", "cand_high")  # score 0.95 -> approves True
        pair_low = ("q_low", "cand_low")    # score 0.20 -> rejects False

        barrier = threading.Barrier(2)

        class InterleavedModel:
            def predict(self, pairs):
                barrier.wait(timeout=5.0)
                pair = tuple(pairs[0])
                if pair == pair_high:
                    return [0.95]
                return [0.20]

        cev = _make_cev(model=InterleavedModel(), threshold=0.85)
        results: dict[str, object] = {}

        def worker_score():
            results["score"] = cev.score(*pair_high)

        def worker_call():
            results["call"] = cev(*pair_low)

        t1 = threading.Thread(target=worker_score)
        t2 = threading.Thread(target=worker_call)

        t1.start()
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert results["score"] == pytest.approx(0.95)
        assert results["call"] is False

    def test_model_exceptions_do_not_poison_cache_and_can_be_retried(self):
        """Model exceptions are not cached and subsequent attempts can succeed."""
        should_fail = True
        lock = threading.Lock()

        class TransientFailureModel:
            def predict(self, pairs):
                nonlocal should_fail
                with lock:
                    if should_fail:
                        raise RuntimeError("transient GPU/network timeout")
                    return [0.91]

        cev = _make_cev(model=TransientFailureModel(), threshold=0.85)
        pair = ("retry_query", "retry_cand")

        # In failure state, score and call fail and do not populate cache
        assert cev.score(*pair) is None
        assert cev(*pair) is False
        assert pair not in cev._cache

        # Transient issue resolves
        with lock:
            should_fail = False

        # Subsequent attempts succeed and populate cache
        score = cev.score(*pair)
        assert score == pytest.approx(0.91)
        assert cev(*pair) is True
        assert pair in cev._cache

    def test_invalid_and_boolean_predictions_not_cached_under_concurrency(self):
        """Boolean and non-finite outputs are rejected and never inserted into cache."""
        num_threads = 6
        barrier = threading.Barrier(num_threads)

        bad_outputs = {
            ("q_bool_true", "c1"): True,
            ("q_bool_false", "c2"): False,
            ("q_np_bool", "c3"): np.bool_(True),
            ("q_nan", "c4"): float("nan"),
            ("q_inf", "c5"): float("inf"),
            ("q_malformed", "c6"): "invalid_str",
        }

        class BadOutputModel:
            def predict(self, pairs):
                barrier.wait(timeout=5.0)
                pair = tuple(pairs[0])
                return [bad_outputs[pair]]

        cev = _make_cev(model=BadOutputModel(), threshold=0.85)

        def worker(pair):
            s = cev.score(*pair)
            ver = cev(*pair)
            return s, ver

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            worker_results = list(executor.map(worker, bad_outputs.keys()))

        for s, ver in worker_results:
            assert s is None
            assert ver is False

        assert len(cev._cache) == 0, "No invalid predictions should be cached"

    def test_cache_eviction_bounded_under_concurrency(self):
        """Cache size remains bounded at max_cache_size under concurrent inserts."""
        max_size = 5
        num_threads = 20
        barrier = threading.Barrier(num_threads)

        class MultiPairModel:
            def predict(self, pairs):
                barrier.wait(timeout=5.0)
                return [0.90]

        cev = _make_cev(model=MultiPairModel(), threshold=0.85, max_cache_size=max_size)

        def worker(i):
            cev.score(f"q_{i}", f"c_{i}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(executor.map(worker, range(num_threads)))

        with cev._cache_lock:
            assert len(cev._cache) <= max_size

    def test_concurrent_semantic_cache_get_isolation(self):
        """Concurrent SemanticCache.get() lookups maintain decision and score isolation."""
        # Query 1 and Candidate 1 -> high score (0.95) -> HIT
        # Query 2 and Candidate 2 -> low score (0.20) -> MISS
        vec_q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec_c1 = np.array([0.98, np.sqrt(1 - 0.98**2), 0.0, 0.0], dtype=np.float32)

        vec_q2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        vec_c2 = np.array([0.0, 0.98, np.sqrt(1 - 0.98**2), 0.0], dtype=np.float32)

        q1_text = "Who is the CEO of Apple?"
        c1_text = "Who is Apple CEO?"  # Guard VERIFY -> verifier high score 0.95

        q2_text = "Who is the CEO of Tesla?"
        c2_text = "Who is Tesla CEO?"  # Guard VERIFY -> verifier low score 0.20

        mapping = {
            q1_text: vec_q1,
            c1_text: vec_c1,
            q2_text: vec_q2,
            c2_text: vec_c2,
        }

        barrier = threading.Barrier(2)

        class PairDifferentiatingModel:
            def predict(self, pairs):
                barrier.wait(timeout=5.0)
                pair = tuple(pairs[0])
                if "Apple" in pair[0]:
                    return [0.95]
                return [0.20]

        cev = _make_cev(model=PairDifferentiatingModel(), threshold=0.85)
        cache = _make_cache(embedder=MockEmbedder(mapping), verifier=cev)

        cache.set(c1_text, "Tim Cook")
        cache.set(c2_text, "Elon Musk")

        get_results: dict[str, object] = {}

        def worker_q1():
            get_results["q1"] = cache.get(q1_text)

        def worker_q2():
            get_results["q2"] = cache.get(q2_text)

        t1 = threading.Thread(target=worker_q1)
        t2 = threading.Thread(target=worker_q2)

        t1.start()
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        # q1 must be a HIT with Tim Cook
        res1 = get_results["q1"]
        assert res1 is not None, "q1 must hit"
        assert res1.hit is True
        assert res1.response == "Tim Cook"

        # q2 must be a MISS (rejected by verifier score 0.20 < 0.85)
        res2 = get_results["q2"]
        assert res2 is None, "q2 must be a miss due to verifier rejection"
