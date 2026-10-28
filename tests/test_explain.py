"""Tests for SemanticCache.explain and POST /explain.

Covers every decision value with a mocked embedder and verifier, plus
read-only guarantees (nothing stored, no LLM call) and 422 on empty input.
"""

from __future__ import annotations

import numpy as np
import pytest

# manual_api_ollama imports ollama + fastapi at module level, which are not
# part of the base install. Skip the whole module cleanly when they are absent
# so CI does not fail at collection time.
pytest.importorskip("ollama")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import manual_api_ollama  # noqa: E402
from manual_api_ollama import app  # noqa: E402
from semantic_cache import HashingEmbedder, SemanticCache, SQLiteStore  # noqa: E402


class _StubEmbedder(HashingEmbedder):
    """HashingEmbedder subclass that accepts OllamaEmbedder kwargs so the
    lifespan can build a cache offline during tests."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()


class _ChatMessage:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content
        self.thinking = None
        self.images = None
        self.tool_name = None
        self.tool_calls = None


class _ChatResponse:
    def __init__(self, content: str) -> None:
        self.model = "llama3.2:3b"
        self.created_at = ""
        self.done = True
        self.done_reason = "stop"
        self.message = _ChatMessage(content)


def _chat_response(content: str) -> _ChatResponse:
    return _ChatResponse(content)


class _FakeOllamaClient:
    """Stand-in for ollama.Client used by the app's lifespan.

    Tests replace ``chat`` via monkeypatch on the instance exposed on
    ``app.state.llm_client`` after startup, so no test ever talks to a real
    Ollama server.
    """

    def chat(self, *args, **kwargs):  # noqa: D401  pragma: no cover
        return _chat_response("")


@pytest.fixture
def client(monkeypatch):
    """TestClient with a stubbed embedder, in-memory store, NO verifier, and a
    fake Ollama chat client — so the lifespan succeeds offline and tests are
    isolated from any real Ollama server."""
    monkeypatch.setattr(manual_api_ollama, "OllamaEmbedder", _StubEmbedder)
    monkeypatch.setattr(manual_api_ollama, "_build_verifier", lambda: None)
    monkeypatch.setattr(
        manual_api_ollama,
        "_build_ollama_client",
        lambda embedder: _FakeOllamaClient(),
    )

    _orig_sqlite_store = manual_api_ollama.SQLiteStore

    class _MemoryStore(_orig_sqlite_store):
        def __init__(self, *args, **kwargs):
            args = (":memory:",)
            kwargs.pop("path", None)
            kwargs.pop("ann_backend", None)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(manual_api_ollama, "SQLiteStore", _MemoryStore)

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# SemanticCache.explain
# ---------------------------------------------------------------------------


def _cache(**kwargs) -> SemanticCache:
    return SemanticCache(
        embedder=HashingEmbedder(),
        store=SQLiteStore(":memory:"),
        threshold=0.85,
        exact_match=False,
        **kwargs,
    )


class TestExplainSemanticCache:
    def test_no_candidates_when_cache_empty(self) -> None:
        cache = _cache()
        result = cache.explain("anything")
        assert result["decision"] == "no_candidates"
        assert result["best_candidate"] is None
        assert result["guard"] is None
        assert result["verifier_score"] is None
        assert result["threshold"] == 0.85

    def test_below_threshold(self) -> None:
        cache = _cache()
        cache.set("What is the capital of France?", "Paris")
        # HashingEmbedder gives near-zero similarity for unrelated text, so
        # explain must not report a hit.
        result = cache.explain("completely unrelated query about boats")
        assert result["decision"] in ("no_candidates", "below_threshold")
        assert result["best_candidate"] is None

    def test_empty_prompt_is_a_miss(self, client) -> None:
        # /explain returns 422 for a whitespace-only prompt (Pydantic) and
        # treats an empty string as a miss, mirroring get().
        response = client.post("/explain", json={"prompt": "   "})
        assert response.status_code == 422

        result = _cache().explain("")
        assert result["decision"] == "no_candidates"
        assert result["best_candidate"] is None


class _SameVectorEmbedder(HashingEmbedder):
    """Returns an identical unit vector for every text, so any pair scores
    1.0 and clears the cosine threshold. Lets the guard/verifier decide."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__(dim=dim)

    def embed(self, text: str) -> np.ndarray:  # noqa: D401
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = 1.0
        return vec


def _same_cache(**kwargs) -> SemanticCache:
    return SemanticCache(
        embedder=_SameVectorEmbedder(),
        store=SQLiteStore(":memory:"),
        threshold=0.85,
        exact_match=False,
        **kwargs,
    )


class TestExplainDecisions:
    def test_guard_reject(self) -> None:
        cache = _same_cache()
        cache.set("Enable the setting", "ok")
        result = cache.explain("Disable the setting")
        assert result["decision"] == "guard_reject"
        assert result["guard"]["verdict"] == "REJECT"
        assert result["best_candidate"]["query"] == "Enable the setting"
        assert result["best_candidate"]["score"] == 1.0

    def test_verifier_reject(self) -> None:
        cache = _same_cache(verifier=lambda _a, _b: False)
        cache.set("Who wrote Hamlet?", "answer")
        result = cache.explain("Who is the author of Hamlet?")
        assert result["decision"] == "verifier_reject"
        assert result["guard"]["verdict"] == "VERIFY"
        assert result["verifier_score"] is None  # plain bool callback exposes no score

    def test_hit_verified(self) -> None:
        cache = _same_cache(verifier=lambda _a, _b: True)
        cache.set("Who wrote Hamlet?", "answer")
        result = cache.explain("Who is the author of Hamlet?")
        assert result["decision"] == "hit_verified"
        assert result["guard"]["verdict"] == "VERIFY"
        assert result["best_candidate"]["query"] == "Who wrote Hamlet?"

    def test_hit_guard_accept(self) -> None:
        cache = _same_cache()
        cache.set("What is the capital of France?", "Paris")
        result = cache.explain("Capital city of France?")
        assert result["decision"] == "hit_guard_accept"
        assert result["guard"]["verdict"] == "ACCEPT"
        assert result["verifier_score"] is None

    def test_hit_exact(self) -> None:
        cache = SemanticCache(
            embedder=_SameVectorEmbedder(),
            store=SQLiteStore(":memory:"),
            threshold=0.85,
            exact_match=True,
        )
        cache.set("What is the capital of France?", "Paris")
        result = cache.explain("What is the capital of France?")
        assert result["decision"] == "hit_exact"
        assert result["best_candidate"]["score"] == 1.0
        assert result["guard"] is None


# ---------------------------------------------------------------------------
# Read-only guarantees
# ---------------------------------------------------------------------------


class TestExplainReadOnly:
    def test_explain_never_stores(self) -> None:
        cache = _same_cache(verifier=lambda _a, _b: True)
        cache.set("Who wrote Hamlet?", "answer")
        before = cache.store.size()
        cache.explain("Who is the author of Hamlet?")
        assert cache.store.size() == before

    def test_post_explain_never_stores(self, client, monkeypatch) -> None:
        """/ask seeds one entry (mocked LLM), then /explain must not add any."""
        monkeypatch.setattr(
            client.app.state.llm_client,
            "chat",
            lambda *a, **k: _chat_response("answer to: " + k["messages"][1]["content"].strip()),
        )
        response = client.post("/ask", json={"prompt": "Who wrote Hamlet?"})
        assert response.status_code == 200
        before = client.app.state.cache.store.size()
        assert before == 1

        # Now /explain must never call the LLM.
        def boom(*a, **k):
            raise AssertionError("LLM must not be called")

        monkeypatch.setattr(client.app.state.llm_client, "chat", boom)
        response = client.post(
            "/explain", json={"prompt": "Who is the author of Hamlet?"}
        )
        assert response.status_code == 200
        assert client.app.state.cache.store.size() == before

    def test_post_explain_never_calls_llm(self, client, monkeypatch) -> None:
        """/explain must succeed without ever invoking the Ollama chat client."""
        def boom(*a, **k):
            raise AssertionError("LLM must not be called")

        monkeypatch.setattr(client.app.state.llm_client, "chat", boom)

        response = client.post(
            "/explain", json={"prompt": "Who is the author of Hamlet?"}
        )
        assert response.status_code == 200
        assert "decision" in response.json()

    def test_post_explain_returns_422_for_empty(self, client) -> None:
        response = client.post("/explain", json={"prompt": ""})
        assert response.status_code == 422

    def test_post_explain_decision_values(self, client) -> None:
        # Empty cache -> no_candidates.
        response = client.post("/explain", json={"prompt": "anything"})
        assert response.status_code == 200
        assert response.json()["decision"] == "no_candidates"
