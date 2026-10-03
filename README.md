# Semantic Cache 🧠⚡

A lightweight, plug-and-play **semantic caching layer** for LLM calls. Instead of matching prompts byte-for-byte, it matches by *meaning* — so near-duplicates and paraphrases return cached answers instantly.

```
"What is the capital of France?"   →  MISS  → calls LLM → cached
"what's the capital of france"     →  HIT   → returned in < 1 ms
"Capital city of France?"          →  HIT   → returned in < 1 ms  ✨
```

---

## Features

| Feature | Details |
|---------|---------|
| 🔌 Plug-and-play | Drop-in decorator for any `fn(prompt) -> str` |
| 🧠 Semantic matching | Cosine similarity on embeddings |
| 💾 SQLite backend | Persistent, zero-server storage |
| ⏰ TTL support | Auto-expire stale entries |
| 🗂️ Namespaces | Isolate caches per model / system prompt / user |
| 📊 Stats | Hit-rate, misses, seconds saved |
| 🔧 Swappable embedders | `HashingEmbedder` (offline) or `SentenceTransformerEmbedder` |

---

## Quick Start

```bash
pip install numpy
# Optional: real semantic embeddings
pip install sentence-transformers
```

```python
from semantic_cache import SemanticCache

cache = SemanticCache(threshold=0.85, ttl=3600)

@cache.cached(namespace="gpt4:v1")
def ask_llm(prompt: str) -> str:
    # your actual LLM call here
    return openai.chat(prompt)

ask_llm("What is the capital of France?")   # MISS → calls LLM
ask_llm("what's the capital of france")      # HIT  → instant
```

---

## Architecture

```
SemanticCache
├── Embedder         (text → unit vector)
│   ├── HashingEmbedder          ← offline, character n-grams
│   └── SentenceTransformerEmbedder ← real paraphrase matching
└── SQLiteStore      (vector index + persistence)
    ├── add()        ← store query + embedding + response
    ├── search()     ← cosine similarity lookup
    └── clear()      ← evict by namespace or all
```

---

## API Reference

### `SemanticCache(threshold, ttl, namespace, embedder, store)`

| Param | Default | Description |
|-------|---------|-------------|
| `threshold` | `0.85` | Minimum cosine similarity to count as a hit |
| `ttl` | `None` | Seconds before entries expire (`None` = never) |
| `namespace` | `"default"` | Partition key (per model / user / prompt version) |
| `embedder` | `HashingEmbedder()` | Any object with `.embed(text) -> np.ndarray` |
| `store` | `SQLiteStore()` | Any compatible store (in-memory by default) |

### Methods

```python
cache.get(query, namespace)              # -> CacheResult | None
cache.set(query, response, namespace)    # store a response
cache.get_or_compute(query, compute_fn)  # fetch or call compute_fn
cache.cached(namespace)(fn)             # decorator
cache.clear(namespace)                   # evict entries
cache.stats                              # Stats(hits, misses, hit_rate, seconds_saved)
```

---

## Running the Demo

```bash
python demo.py
```

---

## License

MIT
