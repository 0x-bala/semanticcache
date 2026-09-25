"""Embedders turn text into unit-length vectors. Swap them freely."""
from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Offline, dependency-light embedder (character n-grams + words, hashed).

    Good for demos, tests and near-duplicate matching. For real paraphrase
    matching, use SentenceTransformerEmbedder or your own API-based embedder.
    """

    def __init__(self, dim: int = 1024, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    def embed(self, text: str) -> np.ndarray:
        text = re.sub(r"[^\w\s]", "", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        vec = np.zeros(self.dim, dtype=np.float32)
        padded = f" {text} "
        features = [padded[i : i + self.ngram] for i in range(len(padded) - self.ngram + 1)]
        features += text.split()
        for f in features:
            vec[zlib.crc32(f.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class SentenceTransformerEmbedder:
    """Real semantic embeddings. Requires: pip install sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)
