"""Tests for OpenAIEmbedder, CohereEmbedder, and OllamaEmbedder."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from semantic_cache.embedders import CohereEmbedder, OllamaEmbedder, OpenAIEmbedder


class _FakeEmbeddingData:
    def __init__(self, embedding: list[float], index: int = 0) -> None:
        self.embedding = embedding
        self.index = index


class _FakeEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingData(vec, idx) for idx, vec in enumerate(embeddings)]


def test_openai_embedder_mocked():
    fake_client = MagicMock()
    vec1 = [0.1] * 1536
    vec2 = [0.2] * 1536

    # Return normalized mock vectors
    def fake_create(model, input, **kwargs):
        if isinstance(input, str):
            return _FakeEmbeddingResponse([vec1])
        else:
            return _FakeEmbeddingResponse([vec1, vec2])

    fake_client.embeddings.create.side_effect = fake_create

    embedder = OpenAIEmbedder(client=fake_client, dimensions=1536)
    v = embedder.embed("Hello world")
    assert isinstance(v, np.ndarray)
    assert v.shape == (1536,)
    assert pytest.approx(float(np.linalg.norm(v)), 1e-4) == 1.0

    batch = embedder.embed_batch(["Hello", "World"])
    assert len(batch) == 2
    assert batch[0].shape == (1536,)
    assert batch[1].shape == (1536,)


def test_openai_embedder_missing_package_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="openai.*required"):
        OpenAIEmbedder(client=None)


def test_cohere_embedder_mocked():
    fake_client = MagicMock()
    vec1 = [0.1] * 1024
    vec2 = [0.2] * 1024

    def fake_embed(texts, model, input_type):
        if len(texts) == 1:
            return [vec1]
        return [vec1, vec2]

    fake_client.embed.side_effect = fake_embed

    embedder = CohereEmbedder(client=fake_client, dimensions=1024)
    v = embedder.embed("Cohere prompt")
    assert isinstance(v, np.ndarray)
    assert v.shape == (1024,)
    assert pytest.approx(float(np.linalg.norm(v)), 1e-4) == 1.0

    batch = embedder.embed_batch(["Cohere prompt 1", "Cohere prompt 2"])
    assert len(batch) == 2
    assert batch[0].shape == (1024,)


def test_cohere_embedder_missing_package_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cohere", None)
    with pytest.raises(ImportError, match="cohere.*required"):
        CohereEmbedder(client=None)


def test_ollama_embedder_mocked():
    embedder = OllamaEmbedder(base_url="http://fake-ollama:11434", dimensions=768)
    vec1 = [0.1] * 768
    vec2 = [0.2] * 768

    with patch.object(embedder, "_post_json") as mock_post:
        # Mock /api/embed response
        mock_post.return_value = {"embeddings": [vec1]}
        v = embedder.embed("Local Ollama test")
        assert isinstance(v, np.ndarray)
        assert v.shape == (768,)
        assert pytest.approx(float(np.linalg.norm(v)), 1e-4) == 1.0

        # Mock /api/embed batch
        mock_post.return_value = {"embeddings": [vec1, vec2]}
        batch = embedder.embed_batch(["Query A", "Query B"])
        assert len(batch) == 2
        assert batch[0].shape == (768,)
        assert batch[1].shape == (768,)
