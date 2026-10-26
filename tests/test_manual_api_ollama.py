"""Tests for manual_api_ollama: chat payload builder, blocklist, and 502 handling."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import manual_api_ollama
from manual_api_ollama import SYSTEM_PROMPT, _build_chat_payload, app
from semantic_cache import HashingEmbedder
from semantic_cache.cache import _default_is_cacheable, _normalize_for_blocklist


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
    """TestClient with a stubbed embedder and in-memory store so the
    lifespan succeeds offline and tests are isolated."""
    monkeypatch.setattr(manual_api_ollama, "OllamaEmbedder", _StubEmbedder)

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


class TestAskEndpoint:
    def test_ask_returns_502_when_ollama_unreachable(self, client, monkeypatch) -> None:
        """/ask must return 502 and store nothing when Ollama is unreachable."""
        import ollama

        def boom(*args, **kwargs):
            raise ollama.RequestError("connection refused")

        monkeypatch.setattr(ollama, "chat", boom)

        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )

        assert response.status_code == 502
        assert "Ollama" in response.json()["detail"]
        assert app.state.cache.store.size() == 0

    def test_ask_returns_502_on_timeout(self, client, monkeypatch) -> None:
        import ollama

        def boom(*args, **kwargs):
            raise TimeoutError("request timed out")

        monkeypatch.setattr(ollama, "chat", boom)

        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )

        assert response.status_code == 502
        assert app.state.cache.store.size() == 0

    def test_ask_returns_502_on_empty_response(self, client, monkeypatch) -> None:
        """An empty Ollama response must surface as 502, never cached."""
        import ollama

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

        def boom(*args, **kwargs):
            return _EmptyResponse()

        monkeypatch.setattr(ollama, "chat", boom)

        response = client.post(
            "/ask",
            json={"prompt": "Who wrote Hamlet?"},
        )

        assert response.status_code == 502
        assert app.state.cache.store.size() == 0

    def test_ask_rejects_whitespace_prompt(self, client, monkeypatch) -> None:
        import ollama

        def boom(*args, **kwargs):
            raise AssertionError("should not be called")

        monkeypatch.setattr(ollama, "chat", boom)

        response = client.post(
            "/ask",
            json={"prompt": "   "},
        )

        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])