"""Tests for OllamaEmbedder keep_alive and 404-only fallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from semantic_cache.embedders import OllamaEmbedder, OllamaNotFoundError


class _FakeEmbedder(OllamaEmbedder):
    """OllamaEmbedder subclass that bypasses the real dimension probe."""

    def __init__(self, post_fn, dimensions: int = 4) -> None:
        self.model_name = "nomic-embed-text"
        self.base_url = "http://localhost:11434"
        self.timeout = 30.0
        self.keep_alive = "30m"
        self.dimensions = dimensions
        self._post_fn = post_fn

    def _post_json(self, endpoint, data):
        return self._post_fn(endpoint, data)


def _ok_response(vec):
    return {"embeddings": [vec]}


# ---------------------------------------------------------------------------
# keep_alive is sent on every request
# ---------------------------------------------------------------------------


class TestKeepAliveSent:
    def test_probe_sends_keep_alive(self) -> None:
        calls = []

        def post(endpoint, data):
            calls.append((endpoint, dict(data)))
            if endpoint == "/api/embed":
                return _ok_response([0.1, 0.2, 0.3, 0.4])
            return {"embedding": [0.1, 0.2, 0.3, 0.4]}

        embedder = _FakeEmbedder(post, dimensions=4)
        embedder._probe_dimensions()
        assert calls[0][0] == "/api/embed"
        assert calls[0][1]["keep_alive"] == "30m"

    def test_embed_sends_keep_alive(self) -> None:
        calls = []

        def post(endpoint, data):
            calls.append((endpoint, dict(data)))
            return _ok_response([0.1, 0.2, 0.3, 0.4])

        embedder = _FakeEmbedder(post, dimensions=4)
        embedder.embed("hello")
        assert calls[0][0] == "/api/embed"
        assert calls[0][1]["keep_alive"] == "30m"
        assert calls[0][1]["input"] == "hello"

    def test_embed_batch_sends_keep_alive(self) -> None:
        calls = []

        def post(endpoint, data):
            calls.append((endpoint, dict(data)))
            # /api/embed returns a list of vectors (one per input).
            return {"embeddings": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]}

        embedder = _FakeEmbedder(post, dimensions=4)
        vectors = embedder.embed_batch(["a", "b"])
        assert calls[0][0] == "/api/embed"
        assert calls[0][1]["keep_alive"] == "30m"
        assert calls[0][1]["input"] == ["a", "b"]
        assert len(vectors) == 2
        assert all(v.shape == (4,) for v in vectors)

    def test_custom_keep_alive_is_used(self) -> None:
        seen = []

        def post(endpoint, data):
            seen.append(data.get("keep_alive"))
            return _ok_response([0.1, 0.2, 0.3, 0.4])

        embedder = _FakeEmbedder(post, dimensions=4)
        embedder.keep_alive = "5m"
        embedder.embed("hello")
        assert seen == ["5m"]


# ---------------------------------------------------------------------------
# 404-only fallback; timeouts and connection errors propagate
# ---------------------------------------------------------------------------


class TestFallbackPolicy:
    def test_404_falls_back_to_legacy(self) -> None:
        calls = []

        def post(endpoint, data):
            calls.append(endpoint)
            if endpoint == "/api/embed":
                raise OllamaNotFoundError("404")
            return {"embedding": [0.1, 0.2, 0.3, 0.4]}

        embedder = _FakeEmbedder(post, dimensions=4)
        vec = embedder.embed("hello")
        assert calls == ["/api/embed", "/api/embeddings"]
        assert vec.shape == (4,)

    def test_timeout_raises_immediately_no_second_attempt(self) -> None:
        calls = []

        def post(endpoint, data):
            calls.append(endpoint)
            raise TimeoutError("timed out")

        embedder = _FakeEmbedder(post, dimensions=4)
        with pytest.raises(TimeoutError):
            embedder.embed("hello")
        assert calls == ["/api/embed"]  # exactly one request

    def test_connection_error_raises_immediately(self) -> None:
        calls = []

        def post(endpoint, data):
            calls.append(endpoint)
            raise ConnectionError("refused")

        embedder = _FakeEmbedder(post, dimensions=4)
        with pytest.raises(ConnectionError):
            embedder.embed("hello")
        assert calls == ["/api/embed"]

    def test_non_404_http_error_raises_immediately(self) -> None:
        calls = []

        def post(endpoint, data):
            calls.append(endpoint)
            raise ConnectionError("HTTP 500")

        embedder = _FakeEmbedder(post, dimensions=4)
        with pytest.raises(ConnectionError):
            embedder.embed("hello")
        assert calls == ["/api/embed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])