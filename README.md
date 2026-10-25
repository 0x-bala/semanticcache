# Semantic Cache 🧠⚡

> High-performance semantic caching layer for LLM calls — match prompts by meaning rather than exact text, eliminate redundant API calls, and cut latency from seconds to under a millisecond with robust guard protection.

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
| 💾 SQLite backend | Persistent, zero-server storage (`SQLiteStore()` with no path is in-memory) |
| ⏰ TTL support | Auto-expire stale entries |
| 🗂️ Namespaces | Isolate caches per model / system prompt / user |
| 📊 Stats | Hit-rate, misses, seconds saved |
| 🔧 Swappable embedders | `HashingEmbedder` (offline) or `SentenceTransformerEmbedder` |
| ⚡ Async API | Native `aget`, `aset`, `aask`, and async streaming caching |
| 🌊 Streaming support | Cache generator and async iterator chunk streams |
| 🏷️ `@cached` decorator | Sync & async function decorator with bypass support |
| 🤝 Adapters | Drop-in integrations for OpenAI, LangChain, and LlamaIndex |
| 🤖 Provider Embedders | OpenAI, Cohere, Ollama, Gemini, SentenceTransformers, Hashing |
| 🧹 Eviction Policies | LRU, LFU (least frequently used), FIFO (first-in first-out) |
| 🚀 In-memory + ANN | Microsecond in-memory index with optional FAISS/HNSW backend |
| 🔐 Context isolation | Opaque namespaces from response-affecting context |

---

## Installation & Extras

```bash
# Core package (zero heavy dependencies)
pip install semantic-cache

# Optional extras:
pip install "semantic-cache[ann]"          # FAISS / HNSW acceleration
pip install "semantic-cache[openai]"       # OpenAI SDK & embedder
pip install "semantic-cache[cohere]"       # Cohere embedder
pip install "semantic-cache[langchain]"    # LangChain cache adapter
pip install "semantic-cache[llama-index]"  # LlamaIndex KVStore adapter
pip install "semantic-cache[all]"          # All optional dependencies
```

---

## Quick Start

```python
from semantic_cache import SemanticCache

cache = SemanticCache(threshold=0.85, ttl=3600)

def mock_llm(prompt: str) -> str:
    return f"Response to: {prompt}"

@cache.cached(namespace="gpt4:v1")
def ask_llm(prompt: str) -> str:
    return mock_llm(prompt)

print(ask_llm("What is the capital of France?"))   # MISS → calls LLM
print(ask_llm("what's the capital of france"))      # HIT  → instant (< 1 ms)
```

---

## Integrations & Advanced Features

### 1. Async API (`aget`, `aset`, `aask`)

Native `async` / `await` support without blocking event loops:

```python
import asyncio
from semantic_cache import SemanticCache

cache = SemanticCache(threshold=0.88)

async def main():
    # Direct async set and get
    await cache.aset("What is async Python?", "Async Python uses coroutines.")
    res = await cache.aget("what is async python")
    if res and res.hit:
        print("Async hit:", res.response)

    # aask: get-or-compute with sync or async LLM call
    async def call_llm(prompt: str) -> str:
        await asyncio.sleep(0.1)
        return f"LLM answer to {prompt}"

    answer = await cache.aask("Explain event loops", call_llm)
    print(answer)

asyncio.run(main())
```

### 2. Streaming Response Caching

Cache chunked streaming responses from generators and async generators:

```python
# Sync streaming
def stream_llm(prompt: str):
    for token in ["Hello ", "world ", "from ", "stream!"]:
        yield token

# First call streams from LLM; subsequent calls replay cached stream instantly
for chunk in cache.get_or_compute_stream("Say hello", stream_llm):
    print(chunk, end="", flush=True)

# Async streaming
async def async_stream_llm(prompt: str):
    for token in ["Async ", "token ", "stream"]:
        yield token

async for chunk in cache.async_get_or_compute_stream("Say hello async", async_stream_llm):
    print(chunk, end="", flush=True)
```

### 3. Standalone `@cached` Decorator

Use the `@cached` decorator on sync and async functions, with support for custom key functions and bypass flags:

```python
from semantic_cache import cached

@cached(threshold=0.90, ttl=3600)
def generate_summary(text: str) -> str:
    return f"Summary of {text}"

# Normal call (hits cache on duplicate/paraphrase)
summary = generate_summary("Analyze quarterly performance")

# Bypass cache when a fresh response is strictly required
fresh = generate_summary("Analyze quarterly performance", bypass=True)
```

### 4. OpenAI Client Wrapper

Wrap standard sync or async `OpenAI` clients to transparently cache completions and streaming responses:

```python
from openai import OpenAI
from semantic_cache import SemanticCache
from semantic_cache.adapters.openai import wrap_openai

client = wrap_openai(OpenAI(), cache=SemanticCache(threshold=0.92))

# Non-streaming call: cached on semantic match
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is Python?"}],
)

# Streaming call: cached and replayed seamlessly
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is Python?"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### 5. LangChain & LlamaIndex Adapters

Drop-in adapters for major AI application frameworks:

```python
# LangChain Integration
from langchain_community.globals import set_llm_cache
from semantic_cache.adapters.langchain import LangChainSemanticCache

set_llm_cache(LangChainSemanticCache(threshold=0.88))

# LlamaIndex Integration
from llama_index.core import Settings
from semantic_cache.adapters.llamaindex import LlamaIndexSemanticCache

kv_cache = LlamaIndexSemanticCache(threshold=0.88)
```

### 6. Provider Embedders (OpenAI, Cohere, Ollama)

Plug in your preferred embedding provider with automatic batching and L2 normalization:

```python
from semantic_cache import SemanticCache
from semantic_cache.embedders import OpenAIEmbedder, CohereEmbedder, OllamaEmbedder

# OpenAI embeddings
cache = SemanticCache(embedder=OpenAIEmbedder(model="text-embedding-3-small"))

# Cohere embeddings
cache = SemanticCache(embedder=CohereEmbedder(model="embed-english-v3.0"))

# Ollama local embeddings (offline / self-hosted)
cache = SemanticCache(embedder=OllamaEmbedder(model="nomic-embed-text"))
```

### 7. Eviction Policies (LFU, FIFO, LRU)

Control cache memory reclamation under `max_entries`:

```python
from semantic_cache import SemanticCache, SQLiteStore

# LFU: evict least frequently used entries first
store_lfu = SQLiteStore("cache.db", max_entries=50000, eviction_policy="lfu")

# FIFO: evict oldest created entries first
store_fifo = SQLiteStore("cache.db", max_entries=50000, eviction_policy="fifo")

# LRU (default): evict least recently accessed entries
store_lru = SQLiteStore("cache.db", max_entries=50000, eviction_policy="lru")

cache = SemanticCache(store=store_lfu)
```

### 8. ANN Index Acceleration (FAISS / HNSW)

For large caches (> 10,000 entries), enable approximate nearest neighbor indexing:

```python
from semantic_cache import SQLiteStore

# Automatically uses FAISS when cache entries exceed ann_threshold (default: 10,000)
store = SQLiteStore("cache.db", ann_backend="faiss", ann_threshold=10000)
```

---

## Architecture

```
SemanticCache
├── Embedder         (text → unit vector)
│   ├── HashingEmbedder          ← offline, character n-grams
│   ├── SentenceTransformerEmbedder ← real paraphrase matching
│   ├── OpenAIEmbedder / CohereEmbedder / OllamaEmbedder
│   └── GeminiEmbedder
└── SQLiteStore      (in-memory index + background persistence)
    ├── Memory Matrix ← < 0.1 ms exact hash & SIMD cosine search
    ├── ANN Backend   ← Optional FAISS / HNSW for large-scale indices
    ├── Batched WAL   ← Non-blocking background persistence
    └── Eviction      ← LRU, LFU, FIFO
```

---

## API Reference

### `SemanticCache(threshold, ttl, namespace, embedder, store, guard, shadow_rate)`

| Param | Default | Description |
|-------|---------|-------------|
| `threshold` | `0.85` | Minimum cosine similarity to count as a hit |
| `ttl` | `None` | Seconds before entries expire (`None` = never) |
| `namespace` | `"default"` | Partition key (per model / user / prompt version) |
| `embedder` | `HashingEmbedder()` | Any embedder implementing `.embed(text)` and `.embed_batch(texts)` |
| `store` | `SQLiteStore()` | In-memory index with batched SQLite persistence |
| `guard` | `True` | Hit guard rejecting semantic collisions (C++/C#, x>5/x<5, km/miles) |
| `shadow_rate` | `0.0` | Sampled background evaluation rate for false-hit monitoring |

### Core Methods

```python
# Synchronous API
cache.get(query, namespace)                      # -> CacheResult | None
cache.set(query, response, namespace)            # Store an entry
cache.get_or_compute(query, compute_fn)          # Get cached or compute
cache.get_or_compute_stream(query, stream_fn)   # Cached streaming
cache.cached(namespace)(fn)                     # Sync/async decorator
cache.clear(namespace)                           # Clear namespace or all

# Asynchronous API
await cache.aget(query, namespace)               # -> CacheResult | None
await cache.aset(query, response, namespace)     # Store entry asynchronously
await cache.aask(query, compute_fn)              # Async get-or-compute
cache.async_get_or_compute_stream(query, fn)     # Async stream generator
```

---

## Measured results (and honest limits)

All numbers below come from `benchmarks/` and are stored in `benchmarks/results_real.json`.
Embeddings are real (`all-MiniLM-L6-v2`, 384 dimensions); nothing here is synthetic.

### Speed at 100,000 cached entries

| | Brute force (NumPy) | FAISS HNSW |
|---|---|---|
| Exact repeat (p50) | 0.005 ms | 0.005 ms |
| Semantic lookup (p50 / p99) | 3.58 ms / 4.10 ms | 0.20 ms / 0.34 ms |
| Miss (p50) | 3.53 ms | 0.34 ms |
| Index build | 10.7 s | 29.6 s |
| Memory (RSS) | 1.13 GB | 1.53 GB |
| FAISS recall@1 / recall@5 vs brute force | n/a | 98.1% / 98.3% |
| Insert past capacity, mean / worst | 0.22 ms / 159 ms | 0.46 ms / 174 ms |

FAISS is about 18x faster for semantic lookups at this size, at the cost of memory and
a small recall loss. Exact repeats are served from a hash table at any size. Below
roughly 10,000 entries brute force is fast enough and needs no extra dependency.

### Accuracy on real labeled pairs (Quora Question Pairs, threshold 0.85)

1,000 true duplicate pairs and 1,000 non-duplicate pairs, decided by the real
`SemanticCache.get()`:

| Setting | Duplicates served (recall) | Wrong answers served (per 1,000 non-duplicates) |
|---|---|---|
| Similarity only (`guard=False`) | 64.0% | 14.7% |
| Guard on, defaults (no verifier) | 7.8% | 0.8% |
| Guard on, `verify_always=False` (less strict) | 21.1% | 2.8% |

What this means:

- The guard cuts wrong answers by roughly 18x compared with pure similarity.
- Without a verifier, ambiguous matches (the guard's `VERIFY` verdict) are treated as
  misses. On loose paraphrases that is most of them, so recall is low. Configure a
  `verifier` (for example `GeminiVerifier`) to recover them.
- QQP is a hard, loosely labeled benchmark: some "non-duplicates" are real paraphrases.
  Traffic where users repeat the same questions will hit far more often.

### Not yet measured

- Accuracy and recall **with** a verifier (live Gemini quota was exhausted; the local
  cross-encoder is uncalibrated).
- `estimated_false_hit_rate` against ground truth with a semantic comparator. A test with
  an exact-string comparator overestimated (45% vs 16% true), so use a semantic comparator.
- A head-to-head against GPTCache or other caches.
- hnswlib at scale (it needs a C++ compiler on Windows). A small correctness test runs
  only when hnswlib is installed.

### Known limits

- Single process: each process keeps its own in-memory index and does not see another
  process's new writes until it reloads.
- Writes are batched (about 50 ms) and flushed on exit; a hard kill can lose the last few
  writes. Use `SQLiteStore(batch_writes=False)` for immediate durability.
- Brute-force search is exact but grows linearly with entries.

### Reproduce

```bash
python tests/benchmark.py                          # offline speed, no key needed
python benchmarks/real_embeddings.py --part 12     # real-embedding speed + accuracy
python benchmarks/guard_rules.py                   # which guard rules block what
```

---

## Verification & Testing Transparency

To ensure complete honesty and clarity regarding performance and testing, here is the breakdown of what has been verified with live external APIs versus offline mocks:

### Verified Live
- **Gemini Embeddings (`gemini-embedding-001`)**:
  - Live API integration with Google Generative Language endpoint verified.
  - Unit normalization of truncated Matryoshka vectors (`norm == 1.000000`) verified live across dimensions 768, 1536, and 3072.
  - Pairwise cosine similarities for the 9 canonical paraphrase and negative test pairs measured live from Google's endpoint at all three dimensions.
- **SQLite Storage & Vector Indexing**:
  - File-based SQLite insert performance measured live on disk across 3,000, 10,000, and 30,000 entries (demonstrating constant amortized ~0.85 ms/op scaling with geometric doubling).
  - Steady-state 10,000-entry full-cache eviction benchmarked live on disk (0.862 ms/op with batch 90% eviction and in-place array compaction, a 162× speedup over the previous single-row deletion and index reload).
  - Multi-process schema migration safety with `BEGIN IMMEDIATE` verified live using two independent OS Python processes executed concurrently against legacy database schemas.
- **Guard & Shadow Mode Execution**:
  - Lexical and semantic entity guards (`check_match`) verified across all candidate pairs.
  - Shadow evaluation thread pool verified to execute asynchronously in the background without adding any delay to user-facing responses.
- **Gemini Verifier Endpoint Network Roundtrip**:
  - Live HTTP calls to `gemini-2.5-flash:generateContent` executed across 99 labeled pairs; measured real roundtrip network latency (p50: 562 ms, p95: 743 ms).

### Verified with Mocks
- **Gemini Text Generation & Verifier Content Parsing**:
  - Text generation (`GeminiClient.generate` and `GeminiVerifier.verify`) requires active Google Cloud `generateContent` quota. When the free tier daily quota (20 requests/day per model) is exhausted, Google returns `HTTP 429: Quota exceeded`.
  - The 5 live LLM generation tests in `pytest` (`test_live_gemini_*` and `test_live_verifier_*`) cleanly skip when encountering HTTP 429 quota exhaustion.
  - LLM response equivalence decision-making was verified using mocked Gemini clients returning deterministic 'yes' / 'no' candidates.

---

## Running the Demo

```bash
python demo.py
```

---

## License

MIT
