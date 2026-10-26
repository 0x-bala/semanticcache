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

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from semantic_cache import (
    OllamaEmbedder,
    SemanticCache,
    SQLiteStore,
)

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise the LLM client, embedder, and cache on startup."""
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


def _generate(prompt: str, model: str) -> str:
    """Call the local Ollama model synchronously via /api/chat.

    Raises ``ConnectionError`` if Ollama returns no usable text so that
    ``/ask`` can convert it to a 502 without caching the failure.
    """
    payload = _build_chat_payload(prompt, model)
    response = ollama.chat(**payload)
    message = getattr(response, "message", None)
    text = getattr(message, "content", "") if message is not None else ""
    if not text:
        raise ConnectionError("Ollama returned an empty response")
    return text


@app.get("/health")
async def health() -> dict[str, Any]:
    cache: SemanticCache = app.state.cache
    return {
        "status": "ok",
        "llm_model": app.state.llm_model,
        "embedder": cache.embedder.__class__.__name__,
        "entries": cache.store.size(),
        "stats": cache.get_stats(),
    }


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    cache: SemanticCache = app.state.cache
    model: str = app.state.llm_model
    try:
        result = await run_in_threadpool(
            cache.get_or_compute,
            request.prompt,
            lambda p: _generate(p, model),
            request.namespace,
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc
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