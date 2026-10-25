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
import zlib
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .guard import DEFAULT_PARAPHRASE_SAFE_WORDS, DEFAULT_STOPWORDS, normalize_text

logger = logging.getLogger(__name__)

# Keep this feature policy aligned with the guard: function/query-framing words
# should not outweigh the informative content words in a short prompt.
_HASHING_NONCONTENT_WORDS = DEFAULT_STOPWORDS | DEFAULT_PARAPHRASE_SAFE_WORDS

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

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


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

    Embedding format v2 weights content-word features more strongly than
    character n-grams. ``cache_version`` changes the cache's embedder key so
    persisted vectors from older feature formats are never compared with v2.
    """

    cache_version = 2

    # Conservative starting point only; calibrate on a held-out labeled dataset.
    default_threshold = 0.90

    def __init__(self, dim: int = 1024, ngram: int = 3) -> None:
        if not (_MIN_DIM <= dim <= _MAX_DIM):
            raise ValueError(f"dim must be between {_MIN_DIM} and {_MAX_DIM}, got {dim}")
        if ngram < 1:
            raise ValueError(f"ngram must be >= 1, got {ngram}")
        self.dim = dim
        self.ngram = ngram

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised float32 vector, or a zero vector for empty input."""
        # Use the same contraction/punctuation normalization as the guard and
        # embedding LRU key. Previously, the LRU could key on "what is ..."
        # while the embedder encoded the raw text "what's ..." differently.
        text = normalize_text(_validate_text(text))

        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            # Empty after normalisation — return zero vector
            return vec

        tokens = text.split()
        padded = f" {text} "
        # Character n-grams provide typo tolerance, but should not dominate
        # paraphrases that retain the same informative words.
        for i in range(len(padded) - self.ngram + 1):
            feature = f"char:{padded[i : i + self.ngram]}"
            vec[zlib.crc32(feature.encode("utf-8")) % self.dim] += 0.25

        for token in tokens:
            word_feature = f"word:{token}"
            vec[zlib.crc32(word_feature.encode("utf-8")) % self.dim] += 1.0

            if token not in _HASHING_NONCONTENT_WORDS:
                content_feature = f"content:{token}"
                vec[zlib.crc32(content_feature.encode("utf-8")) % self.dim] += 2.5

        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 0.0 else vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Return unit-normalised float32 vectors for a batch of texts."""
        return [self.embed(t) for t in texts]


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

    # Starting point only; thresholds vary across models and task domains.
    default_threshold = 0.82

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

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts using SentenceTransformer."""
        for t in texts:
            _validate_text(t)
        v = self.model.encode(texts, normalize_embeddings=True)
        vecs = np.asarray(v, dtype=np.float32)
        return [validate_embedding(vec) for vec in vecs]


# ---------------------------------------------------------------------------
# EmbeddingLRUCache
# ---------------------------------------------------------------------------

class EmbeddingLRUCache:
    """Thread-safe LRU cache for embedding vectors."""

    def __init__(self, capacity: int = 1000) -> None:
        import collections
        import threading

        self.capacity = max(1, capacity)
        self._lock = threading.Lock()
        self._cache: collections.OrderedDict[tuple[str, str], np.ndarray] = (
            collections.OrderedDict()
        )

    def get(self, embedder_name: str, normalized_text: str) -> np.ndarray | None:
        """Retrieve a cached vector or None if not present."""
        key = (embedder_name, normalized_text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key].copy()
        return None

    def put(self, embedder_name: str, normalized_text: str, vec: np.ndarray) -> None:
        """Store a vector in the cache, evicting oldest if capacity reached."""
        key = (embedder_name, normalized_text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)
            self._cache[key] = vec.copy()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        return v / norm
    return v


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------

class OpenAIEmbedder:
    """Embedder using the OpenAI Embeddings API (text-embedding-3-small by default)."""

    default_threshold: float = 0.85

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        client: Any = None,
        dimensions: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        default_dim = 1536 if "small" in model_name or "ada" in model_name else 3072
        self.dimensions = dimensions or default_dim
        if client is not None:
            self.client = client
        else:
            try:
                import openai

                self.client = openai.OpenAI(api_key=api_key, **client_kwargs)
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for OpenAIEmbedder. "
                    "Install it via: pip install 'semantic-cache[openai]'"
                )

    def embed(self, text: str) -> np.ndarray:
        _validate_text(text)
        if not text.strip():
            return np.zeros(self.dimensions, dtype=np.float32)
        kwargs: dict[str, Any] = {"model": self.model_name, "input": text}
        if self.dimensions and "text-embedding-3" in self.model_name:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        vec = _normalize_vec(np.asarray(response.data[0].embedding, dtype=np.float32))
        return validate_embedding(vec, self.dimensions)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        for t in texts:
            _validate_text(t)
        kwargs: dict[str, Any] = {"model": self.model_name, "input": texts}
        if self.dimensions and "text-embedding-3" in self.model_name:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        sorted_data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        return [
            validate_embedding(
                _normalize_vec(np.asarray(item.embedding, dtype=np.float32)), self.dimensions
            )
            for item in sorted_data
        ]


# ---------------------------------------------------------------------------
# CohereEmbedder
# ---------------------------------------------------------------------------

class CohereEmbedder:
    """Embedder using the Cohere Embed API (embed-english-v3.0 by default)."""

    default_threshold: float = 0.85

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        api_key: str | None = None,
        input_type: str = "search_query",
        client: Any = None,
        dimensions: int = 1024,
        **client_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.input_type = input_type
        self.dimensions = dimensions
        if client is not None:
            self.client = client
        else:
            try:
                import cohere

                self.client = cohere.Client(api_key=api_key, **client_kwargs)
            except ImportError:
                raise ImportError(
                    "The 'cohere' package is required for CohereEmbedder. "
                    "Install it via: pip install 'semantic-cache[cohere]'"
                )

    def embed(self, text: str) -> np.ndarray:
        _validate_text(text)
        if not text.strip():
            return np.zeros(self.dimensions, dtype=np.float32)
        response = self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type=self.input_type,
        )
        embeddings = getattr(response, "embeddings", response)
        vec = _normalize_vec(np.asarray(embeddings[0], dtype=np.float32))
        return validate_embedding(vec, self.dimensions)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        for t in texts:
            _validate_text(t)
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type=self.input_type,
        )
        embeddings = getattr(response, "embeddings", response)
        return [
            validate_embedding(
                _normalize_vec(np.asarray(vec, dtype=np.float32)), self.dimensions
            )
            for vec in embeddings
        ]


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """Embedder using a local Ollama instance (nomic-embed-text by default)."""

    default_threshold: float = 0.85

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimensions: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.timeout = timeout

    def _post_json(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        import json
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{endpoint}"
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {exc}"
            ) from exc

    def embed(self, text: str) -> np.ndarray:
        _validate_text(text)
        if not text.strip():
            dim = self.dimensions or 768
            return np.zeros(dim, dtype=np.float32)
        try:
            res = self._post_json("/api/embed", {"model": self.model_name, "input": text})
            embeddings = res.get("embeddings")
            if embeddings:
                raw_vec = embeddings[0]
            else:
                raw_vec = res.get("embedding")
        except Exception:
            res = self._post_json("/api/embeddings", {"model": self.model_name, "prompt": text})
            raw_vec = res.get("embedding")

        if raw_vec is None:
            raise RuntimeError(f"Ollama returned no embedding in response: {res}")
        vec = _normalize_vec(np.asarray(raw_vec, dtype=np.float32))
        if self.dimensions is None:
            self.dimensions = len(vec)
        return validate_embedding(vec, self.dimensions)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        for t in texts:
            _validate_text(t)
        try:
            res = self._post_json("/api/embed", {"model": self.model_name, "input": texts})
            embeddings = res.get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return [
                    validate_embedding(
                        _normalize_vec(np.asarray(vec, dtype=np.float32)), self.dimensions
                    )
                    for vec in embeddings
                ]
        except Exception:
            pass
        return [self.embed(t) for t in texts]


