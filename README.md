# semantic-cache

[![CI](https://github.com/0x-bala/semanticcache/actions/workflows/ci.yml/badge.svg)](https://github.com/0x-bala/semanticcache/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

A semantic cache for LLM applications. It returns a stored answer when a new prompt
means the same thing as an earlier one, and calls the model only when it does not.
A rule-based guard and an optional verifier reduce the chance of returning a wrong answer
for prompts that look alike but are not equivalent.

- Runs in-process with SQLite and NumPy. No server or vector database is required.
- Works with any embedding model: local, hosted, or the built-in offline hashing embedder.
- Sync and async APIs, streaming support, and adapters for OpenAI, LangChain and LlamaIndex.
- Optional FAISS index for large caches.

Contents: [Installation](#installation) | [Quick start](#quick-start) | [Usage](#usage) |
[Architecture](#architecture) | [Performance and accuracy](#performance-and-accuracy) |
[Limitations](#limitations) | [Development](#development) | [Contributing](#contributing) |
[Security](#security) | [License](#license)

## Why a guard is needed

Embedding similarity alone is not a safe test for "same question". These pairs score
above typical thresholds but need different answers:

| Prompt A | Prompt B |
|---|---|
| Convert 10 km to miles | Convert 10 miles to km |
| What is C++? | What is C#? |
| Is x > 5? | Is x < 5? |
| Who is the CEO of Tesla? | Who is the CEO of Apple? |

`semantic-cache` checks each candidate match against explicit rules (numbers, units,
negation, symbols, entities, question type) before returning it. Matches the rules
cannot settle are treated as misses unless a verifier approves them.

## Installation

Requires Python 3.10 or newer. Install from source:

```bash
git clone https://github.com/0x-bala/semanticcache.git
cd semanticcache
pip install -e .
```

The core package depends only on NumPy. Optional extras:

| Extra | Installs | Use |
|---|---|---|
| `ann` | faiss-cpu | Approximate search for large caches |
| `openai` | openai | OpenAI client wrapper and embedder |
| `cohere` | cohere | Cohere embedder |
| `langchain` | langchain-core | LangChain cache adapter |
| `llama-index` | llama-index-core | LlamaIndex adapter |
| `transformers` | sentence-transformers | Local embeddings and cross-encoder verifier |
| `dev` | pytest, ruff, build, anyio | Development |
| `all` | all of the above except `dev` | |

```bash
pip install -e ".[ann,transformers]"
```

## Quick start

```python
from semantic_cache import SemanticCache

cache = SemanticCache(threshold=0.85, ttl=3600)

@cache.cached(namespace="my-app")
def ask(prompt: str) -> str:
    return call_your_llm(prompt)      # replace with a real call

ask("What is the capital of France?")   # miss: calls the function, stores the answer
ask("what's the capital of france")     # hit: answered from the cache
```

The default embedder is an offline character n-gram hasher. It needs no model download
and works for repeated or lightly reworded prompts. For real paraphrase matching, use a
model-based embedder (see [Embedders](#embedders)).

Without the decorator:

```python
result = cache.get("What is the capital of France?")
if result is not None:
    print(result.response, result.score)
else:
    answer = call_your_llm("What is the capital of France?")
    cache.set("What is the capital of France?", answer)
```

### Try it through a local API

The optional FastAPI demo exposes `/ask` and `/health`. It uses Gemini for both
embeddings and text generation, uses FAISS HNSW for similarity search starting
with the first cached entry, and persists entries in `manual_demo_cache.db`.
Install the web and FAISS extras, add `GEMINI_API_KEY=your_gemini_api_key` to a
local `.env` file, and tell Uvicorn to load it:

```powershell
pip install -e ".[web,ann]"
uvicorn manual_api:app --reload --env-file .env
```

Open `http://127.0.0.1:8000/docs` to try the endpoints interactively. Send a prompt
to `POST /ask`; send the same prompt again to see `cache_hit: true`. `/health`
reports cache statistics. This demo has no authentication, so keep it bound to
localhost and do not expose it publicly.

## Usage

### Configuration

```python
SemanticCache(
    embedder=None,        # object with embed(text) and embed_batch(texts); default HashingEmbedder
    store=None,           # default SQLiteStore() in memory
    threshold=0.85,       # minimum cosine similarity for a candidate
    ttl=None,             # seconds before entries expire; None means never
    namespace="default",  # separate caches per model, prompt version or tenant
    guard=True,           # apply the hit guard
    verifier=None,        # callable(new_query, cached_query) -> bool for ambiguous matches
    verify_always=True,   # True: every ambiguous match needs the verifier
    shadow_rate=0.0,      # fraction of hits audited in the background
    exact_match=True,     # serve normalized exact repeats before embedding
)
```

Namespaces keep answers apart. Use a different namespace for each model, system prompt,
or other input that changes the answer.

### Persistence

```python
from semantic_cache import SemanticCache, SQLiteStore

store = SQLiteStore("cache.db", max_entries=50_000, eviction_policy="lru")  # lru | lfu | fifo
cache = SemanticCache(store=store)
...
cache.close()   # flushes pending writes
```

Writes are batched and flushed in the background and on normal exit. Pass
`batch_writes=False` to `SQLiteStore` if every write must be durable immediately.

### Embedders

| Embedder | Notes |
|---|---|
| `HashingEmbedder` | Offline, no dependencies. Good for repeats and small rewordings. |
| `SentenceTransformerEmbedder` | Local model, real paraphrase matching. Needs the `transformers` extra. |
| `OpenAIEmbedder`, `CohereEmbedder`, `OllamaEmbedder` | Hosted or self-hosted providers. |
| `GeminiEmbedder` | Reads `GEMINI_API_KEY`. Output size is configurable (128 to 3072). |

```python
from semantic_cache import SemanticCache, SentenceTransformerEmbedder

cache = SemanticCache(embedder=SentenceTransformerEmbedder(), threshold=0.85)
```

API keys are read from environment variables or passed explicitly. They are never logged.

### The hit guard and verifier

For each candidate above the threshold the guard returns one of three verdicts:

| Verdict | Meaning | Result |
|---|---|---|
| `ACCEPT` | No rule found a difference | Hit |
| `REJECT` | A rule found a conflict (numbers, units, negation, symbols, direction, question type, quoted text, time-sensitive query) | Miss |
| `VERIFY` | Words or roles differ in a way rules cannot settle (different entity in the same slot, swapped roles, different question frame) | Miss, unless a verifier approves |

Without a verifier, `VERIFY` matches are treated as misses. This is the safe default and
it lowers the hit rate on loosely worded paraphrases. To recover them, supply a verifier:

```python
from semantic_cache import SemanticCache, GeminiVerifier

cache = SemanticCache(verifier=GeminiVerifier())    # uses GEMINI_API_KEY
```

Any callable `verifier(new_query, cached_query) -> bool` works, including a local model
such as `CrossEncoderVerifier` (`transformers` extra). A verifier that errors or times out
returns "no", so a failure produces a miss, not a wrong answer.

Pairs the guard always sends to a verifier (swapped roles, a different entity in the same
slot, a yes/no question matched to a how/why question) are verified even when
`verify_always=False`.

### Async and streaming

```python
answer = await cache.aask("Explain event loops", call_llm)       # sync or async callable

for chunk in cache.get_or_compute_stream("Say hello", stream_llm):
    print(chunk, end="")

async for chunk in cache.async_get_or_compute_stream("Say hello", async_stream_llm):
    print(chunk, end="")
```

A cached stream is replayed as a single chunk containing the full response.

### Decorator

```python
from semantic_cache import cached

@cached(ttl=600)
def summarize(prompt: str, model: str = "small") -> str: ...
```

Arguments other than the prompt (here, `model`) are part of the cache key, so calls that
differ only in those arguments never share an answer. Pass `bypass=True` to skip the cache
for one call.

### Framework adapters

```python
# OpenAI
from openai import OpenAI
from semantic_cache import SemanticCache, wrap_openai
client = wrap_openai(OpenAI(), cache=SemanticCache(threshold=0.92))

# LangChain
from langchain_core.globals import set_llm_cache
from semantic_cache import LangChainSemanticCache
set_llm_cache(LangChainSemanticCache())
```

### Monitoring false hits (shadow mode)

Shadow mode re-computes a sample of cache hits with the real model and compares the
results, giving a measured false-hit rate from your own traffic.

```python
cache = SemanticCache(
    shadow_rate=0.05,
    shadow_comparator=lambda fresh, cached: are_equivalent(fresh, cached),
)
...
cache.flush_shadow()
print(cache.stats.estimated_false_hit_rate)
```

The estimate is only as good as the comparator. Comparing exact strings overstates false
hits, because two correct answers rarely match character for character. About 300 audited
hits with no mismatches support a false-hit rate below 1%.

### Large caches (FAISS)

```python
store = SQLiteStore("cache.db", ann_backend="faiss", ann_threshold=10_000)
```

Below `ann_threshold` entries, search is exact brute force. Above it, FAISS HNSW is used
and evicted entries are filtered from results until about a quarter of the index is stale,
at which point the index is rebuilt.

### Tuning the threshold

```bash
python tools/tune_threshold.py --csv pairs.csv
```

The CSV has the columns `query_a,query_b,should_hit`. The tool reports precision and
recall for several thresholds with the guard on and off.

## Architecture

```
SemanticCache
 |-- Exact tier        normalized-text hash -> stored answer
 |-- Embedder          text -> unit vector (with an LRU cache of recent embeddings)
 |-- SQLiteStore
 |     |-- in-memory index    NumPy matrix, optional FAISS HNSW
 |     |-- background writer  batched SQLite writes (WAL mode)
 |     '-- eviction           LRU, LFU or FIFO under max_entries
 |-- Hit guard         ACCEPT / VERIFY / REJECT per candidate
 |-- Verifier          optional, decides VERIFY candidates
 '-- Shadow audit      optional, samples hits against fresh model output
```

A lookup proceeds in this order:

1. Normalize the query and check the exact tier.
2. Embed the query (cached embeddings are reused).
3. Find the nearest stored entries in the namespace at or above the threshold.
4. Run the guard on the best candidate. `REJECT` ends the lookup as a miss.
5. For `VERIFY`, call the verifier if one is configured; otherwise miss.
6. On a hit, record usage for eviction and optionally schedule a shadow audit.

Everything the cache stores lives in one SQLite file. Cross-process safety is limited to
safe concurrent opening of a shared file; see [Limitations](#limitations).

## Performance and accuracy

Numbers come from `benchmarks/results_real.json`. Embeddings are real
(`all-MiniLM-L6-v2`, 384 dimensions) and the questions are from the Quora Question Pairs
dataset. The machine was a 12-thread Windows PC.

### Lookup speed at 100,000 entries

| | Brute force | FAISS HNSW |
|---|---|---|
| Exact repeat (p50) | 0.005 ms | 0.005 ms |
| Semantic lookup (p50 / p99) | 3.58 ms / 4.10 ms | 0.20 ms / 0.34 ms |
| Miss (p50) | 3.53 ms | 0.34 ms |
| Index build | 10.7 s | 29.6 s |
| Memory | 1.13 GB | 1.53 GB |
| Recall@1 against brute force | exact | 98.1% |
| Insert past capacity (mean / worst) | 0.22 ms / 159 ms | 0.46 ms / 174 ms |

Below roughly 10,000 entries brute force is fast enough and needs no extra dependency.

### Accuracy on labeled pairs

1,000 duplicate and 1,000 non-duplicate pairs at threshold 0.85, decided by
`SemanticCache.get()`:

| Setting | Duplicates served | Wrong answers served |
|---|---|---|
| Similarity only (`guard=False`) | 64.0% | 14.7% |
| Guard on, no verifier (default) | 7.8% | 0.8% |
| Guard on, `verify_always=False` | 21.1% | 2.8% |

The guard serves about 18 times fewer wrong answers than similarity alone. Without a
verifier it also serves far fewer correct paraphrases, because ambiguous matches are
treated as misses. The Quora dataset is loosely labeled (some "non-duplicates" are
reasonable paraphrases), so absolute values on your traffic will differ.

Not yet measured: accuracy with a verifier, shadow-mode estimates against ground truth with
a semantic comparator, FAISS on hnswlib at scale, and a comparison with other caches.

To reproduce:

```bash
python tests/benchmark.py                        # offline speed, no keys
python benchmarks/real_embeddings.py --part 12   # real-embedding speed and accuracy
python benchmarks/guard_rules.py                 # which guard rules block which pairs
```

## Limitations

- Single process. Each process keeps its own in-memory index and does not see entries
  written by another process until it reloads.
- Writes are batched (about 50 ms) and flushed on normal exit. A hard kill can lose the
  last few writes.
- Brute-force search is exact but linear in the number of entries.
- The guard is rule-based and English-oriented. It does not understand meaning, which is
  why ambiguous cases are sent to a verifier or treated as misses.
- The database file is not encrypted. Do not cache secrets or personal data without your
  own retention and access controls. See [SECURITY.md](SECURITY.md).

## Development

```bash
pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q -m "not live"
```

Tests that call live APIs are marked `live` and need a key and quota; the rest run offline.
CI runs lint, the tests and a package build on Python 3.10 to 3.13.

Repository layout:

```
semantic_cache/   library code (cache, store, embedders, guard, verifier, adapters)
tests/            unit and regression tests
benchmarks/       reproducible speed and accuracy benchmarks and recorded results
tools/            threshold tuning
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

1. Open an issue to discuss larger changes before writing code.
2. Create a branch, make focused commits, and add tests for any behavior change.
3. Run `ruff check .` and `pytest` locally. Tests must not call the network.
4. Open a pull request that explains what changed and why.

Please report security issues privately through GitHub Security Advisories, not in
public issues.

## Security

Cached responses are stored in plain text. `CacheContext` and namespaces partition data
but are not an authorization mechanism. See [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
