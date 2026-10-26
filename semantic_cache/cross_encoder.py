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

    def __call__(self, query_a: str, query_b: str) -> bool:
        """Return True when the cross-encoder score meets the configured threshold."""
        import numpy as np

        scores = np.asarray(self.model.predict([[query_a, query_b]])).reshape(-1)
        if scores.size != 1 or not np.isfinite(scores[0]):
            return False
        return float(scores[0]) >= self.threshold
