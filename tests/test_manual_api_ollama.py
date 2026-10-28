"""Tests for manual_api_ollama: chat payload builder, blocklist, and 502 handling."""

from __future__ import annotations

import pytest

# manual_api_ollama imports ollama + fastapi at module level, which are not
# part of the base install. Skip the whole module cleanly when they are absent
# so CI does not fail at collection time.
pytest.importorskip("ollama")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import manual_api_ollama  # noqa: E402
from manual_api_ollama import SYSTEM_PROMPT, _build_chat_payload, app  # noqa: E402
from semantic_cache import HashingEmbedder  # noqa: E402
from semantic_cache.cache import (  # noqa: E402
    _default_is_cacheable,
    _normalize_for_blocklist,
)


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


# ---------------------------------------------------------------------------
# is_cacheable blocklist
# ---------------------------------------------------------------------------


class TestIsCacheableBlocklist:
    def test_rejects_i_dont_know(self) -> None:
        assert _default_is_cacheable("I don't know") is False

    def test_rejects_i_dont_know_with_punctuation(self) -> None:
        assert _default_is_cacheable("I don't know.") is False

    def test_rejects_i_dont_know_curly_apostrophe(self) -> None:
        assert _default_is_cacheable("I don\u2019t know") is False

    def test_rejects_i_dont_know_curly_with_punctuation(self) -> None:
        assert _default_is_cacheable("I don\u2019t know.") is False

    def test_rejects_i_dont_know_extra_text(self) -> None:
        assert _default_is_cacheable("I don't know the answer.") is False

    def test_rejects_question_unclear(self) -> None:
        assert _default_is_cacheable("The question is unclear") is False

    def test_rejects_question_unclear_with_punctuation(self) -> None:
        assert _default_is_cacheable("The question is unclear.") is False

    def test_rejects_no_question_provided(self) -> None:
        assert _default_is_cacheable("No question was provided") is False

    def test_rejects_no_question_provided_with_punctuation(self) -> None:
        assert _default_is_cacheable("No question was provided.") is False

    def test_rejects_empty(self) -> None:
        assert _default_is_cacheable("") is False

    def test_rejects_whitespace(self) -> None:
        assert _default_is_cacheable("   ") is False

    def test_accepts_short_legitimate_answer(self) -> None:
        assert _default_is_cacheable("42") is True

    def test_accepts_short_legitimate_answer_with_punctuation(self) -> None:
        assert _default_is_cacheable("Paris.") is True

    def test_accepts_normal_answer(self) -> None:
        assert _default_is_cacheable("The capital of France is Paris.") is True

    def test_normalize_strips_trailing_punctuation(self) -> None:
        assert _normalize_for_blocklist("I don't know.") == "i don't know"
        assert _normalize_for_blocklist("  Paris.!  ") == "paris"


# ---------------------------------------------------------------------------
# Chat payload builder
# ---------------------------------------------------------------------------


class TestChatPayloadBuilder:
    def test_system_prompt_first(self) -> None:
        payload = _build_chat_payload("Who wrote Hamlet?", "llama3.2:3b")
        messages = payload["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] is SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Who wrote Hamlet?"

    def test_strips_user_prompt(self) -> None:
        payload = _build_chat_payload("  What is 2+2?  ", "llama3.2:3b")
        assert payload["messages"][1]["content"] == "What is 2+2?"

    def test_keep_alive_is_top_level(self) -> None:
        payload = _build_chat_payload("hi", "llama3.2:3b")
        assert payload["keep_alive"] == "30m"
        assert "keep_alive" not in payload["options"]

    def test_stream_false(self) -> None:
        payload = _build_chat_payload("hi", "llama3.2:3b")
        assert payload["stream"] is False

    def test_options(self) -> None:
        payload = _build_chat_payload("hi", "llama3.2:3b")
        options = payload["options"]
        assert options == {
            "temperature": 0,
            "seed": 42,
            "num_predict": 120,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }

    def test_model_in_payload(self) -> None:
        payload = _build_chat_payload("hi", "llama3.2:3b")
        assert payload["model"] == "llama3.2:3b"


# ---------------------------------------------------------------------------
# /ask endpoint: 502 on Ollama failure, nothing stored
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """TestClient with a stubbed embedder, in-memory store, and NO verifier,
    so the lifespan succeeds offline and tests are isolated."""
    monkeypatch.setattr(manual_api_ollama, "OllamaEmbedder", _StubEmbedder)
    monkeypatch.setattr(manual_api_ollama, "_build_verifier", lambda: None)

    _orig_sqlite_store = manual_api_ollama.SQLiteStore

    class _MemoryStore(_orig_sqlite_store):
        def __init__(self, *args, **kwargs):
            # Drop any positional path / ann_backend; force an in-memory store.
            args = (":memory:",)
            kwargs.pop("path", None)
            kwargs.pop("ann_backend", None)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(manual_api_ollama, "SQLiteStore", _MemoryStore)

    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_verifier(monkeypatch):
    """Same as ``client`` but with a stubbed verifier present, so /health
    can be checked in both states."""
    monkeypatch.setattr(manual_api_ollama, "OllamaEmbedder", _StubEmbedder)
    monkeypatch.setattr(manual_api_ollama, "_build_verifier", lambda: object())

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


class TestAskEndpoint:
    def _boom_client(self, exc):
        """Return a fake llm_client whose .chat raises *exc*."""

        class _FakeClient:
            def chat(self, *args, **kwargs):
                raise exc

        return _FakeClient()

    def test_ask_returns_502_when_ollama_request_error(self, client, monkeypatch) -> None:
        """/ask must return 502 and store nothing when Ollama raises a RequestError."""
        import ollama

        monkeypatch.setattr(
            app.state, "llm_client", self._boom_client(ollama.RequestError("connection refused"))
        )

        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )

        assert response.status_code == 502
        assert "Ollama" in response.json()["detail"]
        assert app.state.cache.store.size() == 0

    def test_ask_returns_502_on_connection_error(self, client, monkeypatch) -> None:
        """/ask must return 502 and store nothing on a plain ConnectionError."""
        monkeypatch.setattr(
            app.state, "llm_client", self._boom_client(ConnectionError("refused"))
        )

        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )

        assert response.status_code == 502
        assert "unreachable" in response.json()["detail"]
        assert app.state.cache.store.size() == 0

    def test_ask_returns_504_on_httpx_timeout(self, client, monkeypatch) -> None:
        """/ask must return 504 and store nothing on an httpx read timeout."""
        import httpx

        monkeypatch.setattr(
            app.state, "llm_client", self._boom_client(httpx.ReadTimeout("timed out"))
        )

        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )

        assert response.status_code == 504
        assert "timed out" in response.json()["detail"]
        assert app.state.cache.store.size() == 0

    def test_ask_returns_502_on_empty_response(self, client, monkeypatch) -> None:
        """An empty Ollama response must surface as 502, never cached."""
        class _EmptyMessage:
            role = "assistant"
            content = ""
            thinking = None
            images = None
            tool_name = None
            tool_calls = None

        class _EmptyResponse:
            model = "llama3.2:3b"
            created_at = ""
            done = True
            done_reason = "stop"
            message = _EmptyMessage()

        class _FakeClient:
            def chat(self, *args, **kwargs):
                return _EmptyResponse()

        monkeypatch.setattr(app.state, "llm_client", _FakeClient())

        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )

        assert response.status_code == 502
        assert app.state.cache.store.size() == 0

    def test_ask_rejects_whitespace_prompt(self, client, monkeypatch) -> None:
        class _FakeClient:
            def chat(self, *args, **kwargs):
                raise AssertionError("should not be called")

        monkeypatch.setattr(app.state, "llm_client", _FakeClient())

        response = client.post(
            "/ask",
            json={"prompt": "   "},
        )

        assert response.status_code == 422

    def test_ask_normal_answer_still_works(self, client, monkeypatch) -> None:
        """/ask must return 200 and cache the answer on a normal Ollama reply."""
        class _Message:
            role = "assistant"
            content = "Hamlet was written by William Shakespeare."

        class _Response:
            model = "llama3.2:3b"
            created_at = ""
            done = True
            done_reason = "stop"
            message = _Message()

        class _FakeClient:
            def chat(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(app.state, "llm_client", _FakeClient())

        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["response"] == "Hamlet was written by William Shakespeare."
        assert body["cache_hit"] is False
        assert app.state.cache.store.size() == 1

        # Second call must be served from cache (no second chat call).
        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )
        assert response.status_code == 200
        assert response.json()["cache_hit"] is True
        assert app.state.cache.store.size() == 1


class TestLifespanFailsWhenOllamaDown:
    """When OllamaEmbedder can't reach the server, the lifespan must fail
    with a clear RuntimeError rather than silently starting."""

    def test_lifespan_raises_when_embedder_probe_fails(self, monkeypatch) -> None:
        from semantic_cache.embedders import OllamaEmbedder

        def boom(*args, **kwargs):
            raise ConnectionError("refused")

        monkeypatch.setattr(OllamaEmbedder, "_probe_dimensions", boom)

        with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
            with TestClient(app) as client:
                client.get("/health")


class TestHealthVerifierField:
    def test_health_verifier_false_when_absent(self, client) -> None:
        """/health must report verifier=false when no verifier is configured."""
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["verifier"] is False
        assert "stats" in body
        assert "verify_calls" in body["stats"]
        assert "verify_accepts" in body["stats"]

    def test_health_verifier_true_when_present(self, client_with_verifier) -> None:
        """/health must report verifier=true when a verifier is configured."""
        response = client_with_verifier.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["verifier"] is True
        assert "stats" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])