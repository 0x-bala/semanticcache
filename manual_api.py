"""Local FastAPI app for manually testing the semantic cache with Gemini."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from semantic_cache import (
    GeminiAPIError,
    GeminiClient,
    GeminiEmbedder,
    SemanticCache,
    SQLiteStore,
)

FAISS_ANN_THRESHOLD = 1


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    embedder = GeminiEmbedder()
    app.state.llm = GeminiClient()
    store = SQLiteStore(
        "manual_demo_cache.db",
        ann_backend="faiss",
        ann_threshold=FAISS_ANN_THRESHOLD,
    )
    app.state.cache = SemanticCache(
        embedder=embedder,
        store=store,
        threshold=0.88,
    )
    try:
        yield
    finally:
        app.state.cache.close(close_store=True)


app = FastAPI(
    title="Semantic Cache Manual Test API",
    description="A local demo API for trying semantic-cache with Gemini.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_768)
    namespace: str = Field(default="manual-demo", min_length=1, max_length=256)


@app.get("/health")
async def health() -> dict[str, Any]:
    cache: SemanticCache = app.state.cache
    return {
        "status": "ok",
        "entries": cache.store.size(),
        "stats": cache.get_stats(),
    }


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    cache: SemanticCache = app.state.cache
    llm: GeminiClient = app.state.llm
    try:
        result = await run_in_threadpool(
            cache.get_or_compute,
            request.prompt,
            llm.generate,
            request.namespace,
        )
    except GeminiAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "response": result.response,
        "cache_hit": result.hit,
        "similarity": result.score,
        "matched_query": result.matched_query,
        "namespace": request.namespace,
    }
