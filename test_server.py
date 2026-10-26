"""Standalone test server mirroring manual_api_ollama /ask.

Configurable via env vars so the live semantic test can run guard ON and
guard OFF against fresh databases without touching manual_api_ollama.py.

Env:
  TEST_GUARD      "true" | "false"   (default true)
  TEST_DB_PATH    path to SQLite DB   (default :memory:)
  TEST_PORT       uvicorn port        (default 8766)
"""

from __future__ import annotations

import os

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from semantic_cache import OllamaEmbedder, SemanticCache, SQLiteStore

GUARD = os.environ.get("TEST_GUARD", "true").strip().lower() in {"1", "true", "yes", "on"}
DB_PATH = os.environ.get("TEST_DB_PATH", ":memory:")
PORT = int(os.environ.get("TEST_PORT", "8766"))

LLM_MODEL = "llama3.2:3b"
EMBEDDER_MODEL = "nomic-embed-text"
CACHE_THRESHOLD = 0.85

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


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_768)
    namespace: str = Field(default="semantic-test", min_length=1, max_length=256)

    @field_validator("prompt")
    @classmethod
    def _reject_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must contain at least one non-whitespace character")
        return v


def _generate(prompt: str, model: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ],
        stream=False,
        keep_alive="30m",
        options={
            "temperature": 0,
            "seed": 42,
            "num_predict": 120,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    )
    text = getattr(response.message, "content", "")
    if not text:
        raise ConnectionError("Ollama returned an empty response")
    return text


embedder = OllamaEmbedder(model_name=EMBEDDER_MODEL)
store = SQLiteStore(DB_PATH, embedding_dim=embedder.dimensions)
cache = SemanticCache(
    embedder=embedder,
    store=store,
    threshold=CACHE_THRESHOLD,
    guard=GUARD,
)

app = FastAPI(title=f"Semantic test server (guard={GUARD})")


@app.post("/ask")
async def ask(request: AskRequest) -> dict:
    try:
        result = await run_in_threadpool(
            cache.get_or_compute,
            request.prompt,
            lambda p: _generate(p, LLM_MODEL),
            request.namespace,
        )
    except (ollama.RequestError, ollama.ResponseError, ConnectionError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "response": result.response,
        "cache_hit": result.hit,
        "similarity": result.score,
        "matched_query": result.matched_query,
        "namespace": request.namespace,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "guard": GUARD, "entries": cache.store.size()}