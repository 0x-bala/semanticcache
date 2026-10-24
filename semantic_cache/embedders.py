"""Embedders: convert text to unit-length float32 vectors.

Each embedder must satisfy the Embedder protocol:
  embed(text: str) -> np.ndarray  (float32, unit-norm or zero vector)

Embedding similarity cannot guarantee semantic equivalence.
High cosine similarity means *structurally* similar embeddings, not
necessarily the same *meaning*. Always set thresholds conservatively.

Security note:
  Do not log or expose raw embeddings in production — they may allow
  partial reconstruction of the original text.
"""
from __future__ import annotations

import logging
import re
import zlib
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_DIM = 4
_MAX_DIM = 65_536
_MAX_TEXT_BYTES = 32_768  # 32 KB — prevents accidental huge embeddings


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_text(text: str) -> str:
    """Validate and coerce query text.  Raises ValueError on bad input."""
    if not isinstance(text, str):
        raise TypeError(f"query must be str, got {type(text).__name__}")
    if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(
            f"query exceeds maximum allowed size of {_MAX_TEXT_BYTES} bytes"
        )
    return text


def validate_embedding(vec: np.ndarray, expected_dim: int | None = None) -> np.ndarray:
    """Validate a float32 embedding vector.

    Checks:
    - correct ndarray type
    - 1-D shape
    - float32 dtype (coerced if safe)
    - no NaN or Inf values
    - optional dimension match

    Returns a validated (possibly coerced) copy.
    Raises ValueError / TypeError on unrecoverable problems.
    """
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"embedding must be np.ndarray, got {type(vec).__name__}")
    if vec.ndim != 1:
        raise ValueError(f"embedding must be 1-D, got shape {vec.shape}")
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    if not np.all(np.isfinite(vec)):
        raise ValueError("embedding contains NaN or Inf values")
    if expected_dim is not None and vec.shape[0] != expected_dim:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dim}, got {vec.shape[0]}"
        )
    return vec


def is_zero_vector(vec: np.ndarray) -> bool:
    """Return True if the vector has zero norm (cannot be used for similarity)."""
    return float(np.linalg.norm(vec)) == 0.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    """Duck-type protocol for any embedder.

    Contract:
    - Returns a 1-D float32 np.ndarray.
    - The vector is unit-normalised **or** all-zero (for empty / unembeddable input).
    - Must not raise for valid UTF-8 strings.
    - Must be thread-safe (or documented if not).
    """

    def embed(self, text: str) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# HashingEmbedder
# ---------------------------------------------------------------------------

class HashingEmbedder:
    """Offline, dependency-light embedder (character n-grams + words, hashed).

    Good for demos, tests and near-duplicate detection.  For real paraphrase
    matching, use SentenceTransformerEmbedder or an API-based embedder.

    Thread safety: stateless after construction — safe to share across threads.

    Limitations:
    - Lexically similar but semantically opposite sentences (e.g., "good" vs
      "not good") may score high.  Do not rely on this embedder for subtle
      semantic differences.
    - Empty or whitespace-only input returns a zero vector.  Callers that
      receive a zero vector should treat the result as a cache miss.
    """

    def __init__(self, dim: int = 1024, ngram: int = 3) -> None:
        if not (_MIN_DIM <= dim <= _MAX_DIM):
            raise ValueError(f"dim must be between {_MIN_DIM} and {_MAX_DIM}, got {dim}")
        if ngram < 1:
            raise ValueError(f"ngram must be >= 1, got {ngram}")
        self.dim = dim
        self.ngram = ngram

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised float32 vector, or a zero vector for empty input."""
        _validate_text(text)
        text = re.sub(r"[^\w\s]", "", text.lower())
        text = re.sub(r"\s+", " ", text).strip()

        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            # Empty after normalisation — return zero vector
            return vec

        padded = f" {text} "
        features: list[str] = [
            padded[i : i + self.ngram]
            for i in range(len(padded) - self.ngram + 1)
        ]
        features += text.split()

        for f in features:
            # zlib.crc32 always returns unsigned int in Python 3
            vec[zlib.crc32(f.encode("utf-8")) % self.dim] += 1.0

        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 0.0 else vec


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """Real semantic embeddings using a sentence-transformers model.

    Requires:
        pip install sentence-transformers

    Thread safety: the underlying SentenceTransformer model is NOT guaranteed
    thread-safe.  Use one instance per thread, or protect with a lock.

    Limitations:
    - Embedding similarity does not guarantee answer equivalence.
    - Model quality varies; always validate on your actual query distribution.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        _validate_text(text)
        v = self.model.encode(text, normalize_embeddings=True)
        vec = np.asarray(v, dtype=np.float32)
        # Validate before returning
        return validate_embedding(vec)
