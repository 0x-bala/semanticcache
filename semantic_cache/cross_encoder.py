"""Optional local cross-encoder verifier for ambiguous cache matches.

This verifier is opt-in and requires ``sentence-transformers``. The default
model is an STS regression model whose scores are on a **0..1 scale** (the
library's default threshold is 0.9, calibrated on the QQP benchmark: at 0.9,
about 31% of VERIFY duplicates are recovered while 2.5% of non-duplicates are
accepted). A score is a similarity signal, not proof that two prompts have
interchangeable answers; calibrate ``threshold`` on a task-specific labeled
dataset.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any


class CrossEncoderVerifier:
    """Callable verifier compatible with ``SemanticCache(verifier=...)``.

    Parameters
    ----------
    model_name:
        Sentence-Transformers cross-encoder model. The default is trained for
        semantic textual similarity and returns scores on a 0..1 scale, so
        ``threshold`` must be in that range.
    threshold:
        Minimum predicted score required to accept a pair (default 0.9).
        This is a starting point and must be calibrated for your task.
    device:
        Optional Sentence-Transformers device such as ``"cpu"`` or ``"cuda"``.
    max_length:
        Maximum token length for each query pair.
    max_cache_size:
        Maximum number of scores retained in the thread-safe LRU cache (default 1000).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/stsb-roberta-base",
        threshold: float = 0.9,
        device: str | None = None,
        max_length: int = 256,
        max_cache_size: int = 1000,
    ) -> None:
        if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0 for the default STS model")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 8:
            raise ValueError("max_length must be an integer >= 8")
        if (
            isinstance(max_cache_size, bool)
            or not isinstance(max_cache_size, int)
            or max_cache_size < 1
        ):
            raise ValueError("max_cache_size must be an integer >= 1")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderVerifier requires sentence-transformers. "
                "Install it with: pip install 'semantic-cache[transformers]'"
            ) from exc
        kwargs: dict[str, Any] = {"max_length": max_length}
        if device is not None:
            kwargs["device"] = device
        self.model_name = model_name
        self.threshold = float(threshold)
        self.model = CrossEncoder(model_name, **kwargs)
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._inflight: dict[tuple[str, str], threading.Event] = {}
        self._last_pair: tuple[str, str] | None = None
        self._last_score: float | None = None

    def _compute_score(self, query_a: str, query_b: str) -> float | None:
        """Run model inference and return a valid, finite numeric float, or None."""
        import numpy as np

        try:
            scores = np.asarray(self.model.predict([[query_a, query_b]])).reshape(-1)
        except Exception:
            return None

        if scores.size != 1:
            return None

        raw_val = scores[0]
        # Reject Python boolean and NumPy boolean scores
        if isinstance(raw_val, (bool, np.bool_)) or scores.dtype == bool:
            return None

        if not (np.issubdtype(scores.dtype, np.number) or isinstance(raw_val, (int, float))):
            return None

        try:
            val = float(raw_val)
        except Exception:
            return None

        if not math.isfinite(val):
            return None

        return val

    def score(self, query_a: str, query_b: str) -> float | None:
        """Return the raw predicted similarity score, or None if invalid/non-finite.

        Thread-safe: cached scores are keyed by exact ordered query pair in a
        bounded LRU cache. Concurrent requests for the same pair share in-flight
        inference, while requests for distinct pairs compute concurrently.
        """
        key = (query_a, query_b)
        lock = getattr(self, "_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._cache_lock = lock

        while True:
            with lock:
                if not hasattr(self, "_cache"):
                    self._cache = OrderedDict()
                if not hasattr(self, "_inflight"):
                    self._inflight = {}

                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]

                if key in self._inflight:
                    event = self._inflight[key]
                else:
                    event = threading.Event()
                    self._inflight[key] = event
                    break

            event.wait()

        val: float | None = None
        try:
            val = self._compute_score(query_a, query_b)
            return val
        finally:
            with lock:
                if val is not None:
                    self._cache[key] = val
                    max_size = getattr(self, "max_cache_size", 1000)
                    if len(self._cache) > max_size:
                        self._cache.popitem(last=False)
                    self._last_pair = key
                    self._last_score = val
                self._inflight.pop(key, None)
                event.set()

    def clear_cache(self) -> None:
        """Clear the cached scores."""
        lock = getattr(self, "_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._cache_lock = lock
        with lock:
            if hasattr(self, "_cache"):
                self._cache.clear()
            self._last_pair = None
            self._last_score = None

    def __call__(self, query_a: str, query_b: str) -> bool:
        """Return True when the cross-encoder score meets the configured threshold."""
        val = self.score(query_a, query_b)
        if val is None:
            return False
        return val >= self.threshold
