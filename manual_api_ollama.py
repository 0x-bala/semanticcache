"""Local FastAPI app for manually testing the semantic cache with a local Ollama LLM.

Fully local — no API keys, no network calls beyond the Ollama server.

Setup:
    1. Install Ollama (https://ollama.com)
    2. ollama pull llama3.2:3b        # or :8b if your GPU has 8 GB VRAM
    3. ollama pull nomic-embed-text   # for embeddings
       (or skip this and use SentenceTransformerEmbedder instead)
    4. pip install -e ".[web,ann]"

Run:
    uvicorn manual_api_ollama:app --reload
    Open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

try:
    import ollama
except ImportError:
    raise ImportError(
        "The ollama package is required for this demo. "
        "Install it with: pip install -e '.[ollama]'"
    )

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from semantic_cache import (
    OllamaEmbedder,
    SemanticCache,
    SQLiteStore,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — change these to match your hardware and preference
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You answer factual questions for a cache. Reply in plain text only.

- Start with the answer. No greeting, no filler.
- Use one or two short sentences.
- Name the subject in full. Never say "it" or "this".
- Do not use words like "today", "now" or "currently".
- If you don't know, say: I don't know.
- If the question is unclear, say: The question is unclear.
- Never add extra offers of help or follow-up questions.

Example:
Question: Who wrote Hamlet?
Answer: Hamlet was written by William Shakespeare."""

LLM_MODEL = "llama3.2:3b"          # 3B fits 4 GB VRAM; use :8b for 8 GB cards
EMBEDDER_MODEL = "nomic-embed-text"  # Ollama embedding model
FAISS_ANN_THRESHOLD = 10_000        # switch to FAISS HNSW above 10k entries
CACHE_THRESHOLD = 0.85              # minimum cosine similarity for a hit
CACHE_TTL = 3600                    # 1 hour TTL
VERIFIER_THRESHOLD = 0.9            # cross-encoder score floor (0..1 scale)
OLLAMA_TIMEOUT_S = 60               # request timeout for the Ollama chat client


def _build_verifier():
    """Build the CrossEncoderVerifier, or None if sentence-transformers is absent."""
    try:
        from semantic_cache.cross_encoder import CrossEncoderVerifier
    except ImportError as exc:
        logger.warning(
            "sentence-transformers not installed; running without a verifier. "
            "Install it with: pip install -e '.[transformers]'. (%s)",
            exc,
        )
        return None
    try:
        return CrossEncoderVerifier(threshold=VERIFIER_THRESHOLD)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not initialise CrossEncoderVerifier: %s", exc)
        return None


def _build_ollama_client(embedder: Any) -> Any:
    """Build a single Ollama chat client bound to the embedder's base URL."""
    import ollama

    host = getattr(embedder, "base_url", None) or "http://127.0.0.1:11434"
    return ollama.Client(host=host, timeout=OLLAMA_TIMEOUT_S)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise the LLM client, embedder, cache, and verifier on startup."""
    app.state.llm_model = LLM_MODEL

    try:
        embedder = OllamaEmbedder(model_name=EMBEDDER_MODEL)
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at startup for embeddings "
            f"(model={EMBEDDER_MODEL!r}): {exc}. "
            "Start the Ollama server and pull the model: "
            f"ollama pull {EMBEDDER_MODEL}"
        ) from exc

    verifier = _build_verifier()
    app.state.verifier = verifier
    app.state.llm_client = _build_ollama_client(embedder)

    store = SQLiteStore(
        "manual_ollama_cache.db",
        ann_backend="faiss",
        ann_threshold=FAISS_ANN_THRESHOLD,
    )
    app.state.cache = SemanticCache(
        embedder=embedder,
        store=store,
        threshold=CACHE_THRESHOLD,
        ttl=CACHE_TTL,
        verifier=verifier,
        allow_question_intent_mismatch=True,
    )
    try:
        yield
    finally:
        app.state.cache.close(close_store=True)


app = FastAPI(
    title="Semantic Cache Manual Test API (Ollama)",
    description="A local demo API for trying semantic-cache with a local Ollama LLM.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_768)
    namespace: str = Field(default="ollama-demo", min_length=1, max_length=256)

    @field_validator("prompt")
    @classmethod
    def _reject_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must contain at least one non-whitespace character")
        return v


def _build_chat_payload(prompt: str, model: str) -> dict[str, Any]:
    """Build the /api/chat payload: system prompt first, then the user message.

    keep_alive is a top-level field (not inside options).
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 120,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }


def _generate(prompt: str, model: str, client: Any) -> str:
    """Call the local Ollama model synchronously via /api/chat.

    Raises ``ConnectionError`` if Ollama returns no usable text so that
    ``/ask`` can convert it to a 502 without caching the failure. Raises
    ``httpx.TimeoutException`` on read/connect timeouts so ``/ask`` can
    return 504.
    """
    payload = _build_chat_payload(prompt, model)
    response = client.chat(**payload)
    message = getattr(response, "message", None)
    text = getattr(message, "content", "") if message is not None else ""
    if not text:
        raise ConnectionError("Ollama returned an empty response")
    return text


@app.get("/health")
async def health() -> dict[str, Any]:
    cache: SemanticCache = app.state.cache
    stats = cache.get_stats()
    return {
        "status": "ok",
        "llm_model": app.state.llm_model,
        "embedder": cache.embedder.__class__.__name__,
        "verifier": app.state.verifier is not None,
        "allow_question_intent_mismatch": cache.allow_question_intent_mismatch,
        "entries": cache.store.size(),
        "stats": stats,
    }


@app.post("/explain")
async def explain(request: AskRequest) -> dict[str, Any]:
    """Describe how the cache would resolve *request.prompt*.

    Read-only: no LLM call, nothing stored. Returns the top candidate, the
    effective cosine threshold, the decision, the guard verdict/reason (if the
    guard ran), and the cross-encoder score (if the verifier ran).
    """
    cache: SemanticCache = app.state.cache
    try:
        return await run_in_threadpool(
            cache.explain, request.prompt, namespace=request.namespace
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    cache: SemanticCache = app.state.cache
    model: str = app.state.llm_model
    client = app.state.llm_client
    try:
        result = await run_in_threadpool(
            cache.get_or_compute,
            request.prompt,
            lambda p: _generate(p, model, client),
            request.namespace,
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"Ollama timed out: {exc}") from exc
    except (ConnectionError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "response": result.response,
        "cache_hit": result.hit,
        "similarity": result.score,
        "matched_query": result.matched_query,
        "namespace": request.namespace,
    }