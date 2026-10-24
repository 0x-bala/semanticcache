"""semantic_cache — semantic caching layer for LLM calls.

Public API (backward-compatible with v0.1.0):
    SemanticCache, CacheResult, Stats, SQLiteStore,
    HashingEmbedder, SentenceTransformerEmbedder

New in v0.2.0:
    SemanticCache.enabled        — bypass flag
    SemanticCache.delete()       — remove exact entries
    SemanticCache.invalidate_expired() — prune TTL-expired entries
    SemanticCache.get_stats() / reset_stats()
    SemanticCache.namespace_stats()
    SQLiteStore.invalidate_expired()
    SQLiteStore.delete_by_query()
    SQLiteStore.namespace_stats()
    embedders.validate_embedding()
    embedders.is_zero_vector()
"""
from .cache import CacheResult, SemanticCache, Stats
from .context import CacheContext
from .embedders import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    is_zero_vector,
    validate_embedding,
)
from .gemini import GeminiAPIError, GeminiClient, GeminiEmbedder
from .protocols import CacheStore
from .store import SQLiteStore

__version__ = "0.2.0"

__all__ = [
    # Cache
    "SemanticCache",
    "CacheResult",
    "Stats",
    "CacheContext",
    # Store
    "SQLiteStore",
    "CacheStore",
    # Embedders
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "GeminiClient",
    "GeminiEmbedder",
    "GeminiAPIError",
    # Helpers
    "validate_embedding",
    "is_zero_vector",
]
