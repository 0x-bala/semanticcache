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
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/stsb-roberta-base",
        threshold: float = 0.9,
        device: str | None = None,
        max_length: int = 256,
    ) -> None:
        if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0 for the default STS model")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 8:
            raise ValueError("max_length must be an integer >= 8")
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
        self._last_pair: tuple[str, str] | None = None
        self._last_score: float | None = None

    def score(self, query_a: str, query_b: str) -> float | None:
        """Return the raw predicted similarity score, or None if invalid/non-finite."""
        if self._last_pair == (query_a, query_b):
            return self._last_score

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

        if not np.issubdtype(scores.dtype, np.number) and not isinstance(raw_val, (int, float)):
            return None

        try:
            val = float(raw_val)
        except Exception:
            return None

        if not math.isfinite(val):
            return None

        self._last_pair = (query_a, query_b)
        self._last_score = val
        return val

    def __call__(self, query_a: str, query_b: str) -> bool:
        """Return True when the cross-encoder score meets the configured threshold."""
        val = self.score(query_a, query_b)
        if val is None:
            return False
        return val >= self.threshold
