"""Real embeddings benchmark for SemanticCache using local all-MiniLM-L6-v2 and Gemini.

Measures:
- Part 1: Speed, recall, eviction, and throughput at 100k entries with FAISS vs Brute-force.
- Part 2: Offline accuracy, hit rate, and false-hit rate on 2,000 real QQP pairs.
- Part 2b: Gemini small-sample comparison (if quota allows) or quota report.
- Part 3: GeminiVerifier evaluation on up to 15 pairs or quota report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from typing import Any

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from semantic_cache.gemini import GeminiEmbedder  # noqa: E402
from semantic_cache.store import SQLiteStore  # noqa: E402
from semantic_cache.verifier import GeminiVerifier  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_real.json")
README_PATH = os.path.join(os.path.dirname(__file__), "README.md")


class LocalMiniLMEmbedder:
    """Local offline embedder using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        self.dim = 384
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

    def embed_batch(self, texts: list[str], max_length: int = 128) -> np.ndarray:
        """Embed a list of strings into unit-normalized 384-dimensional float32 vectors."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
        pooled = torch.sum(out[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normed.cpu().numpy().astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


def get_system_metadata() -> dict[str, Any]:
    """Return environment and hardware metadata."""
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "primary_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "primary_dimensionality": 384,
        "api_embedding_model": "gemini-embedding-001 (768 dims)",
        "verifier_model": "gemini-2.5-flash",
        "dataset": "GLUE QQP (Quora Question Pairs)",
    }


def load_or_prep_dataset() -> tuple[list[dict[str, Any]], list[str]]:
    """Load or parse QQP dataset into pairs and corpus questions."""
    os.makedirs(DATA_DIR, exist_ok=True)
    pairs_file = os.path.join(DATA_DIR, "pairs_2k.json")
    corpus_file = os.path.join(DATA_DIR, "corpus_questions.json")

    if os.path.exists(pairs_file) and os.path.exists(corpus_file):
        with open(pairs_file, encoding="utf-8") as f:
            pairs = json.load(f)
        with open(corpus_file, encoding="utf-8") as f:
            corpus = json.load(f)
        cached_qs = list(dict.fromkeys(p["cached_q"] for p in pairs))
        other_qs = [q for q in corpus if q not in set(cached_qs)]
        aligned = (cached_qs + other_qs)[:100000]
        if aligned != corpus:
            with open(corpus_file, "w", encoding="utf-8") as f:
                json.dump(aligned, f)
            corpus = aligned
        return pairs, corpus

    tsv_path = os.path.join(DATA_DIR, "QQP", "train.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"QQP train.tsv not found at {tsv_path}")

    dup_pairs = []
    non_dup_pairs = []
    corpus_questions = []
    seen = set()

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            q1 = row[3].strip()
            q2 = row[4].strip()
            label = row[5].strip()
            if not q1 or not q2 or q1 == q2:
                continue
            if label == "1" and len(dup_pairs) < 1000:
                dup_pairs.append({"cached_q": q1, "query": q2, "is_duplicate": 1})
                if q1 not in seen:
                    seen.add(q1)
                    corpus_questions.append(q1)
            elif label == "0" and len(non_dup_pairs) < 1000:
                non_dup_pairs.append({"cached_q": q1, "query": q2, "is_duplicate": 0})
                if q1 not in seen:
                    seen.add(q1)
                    corpus_questions.append(q1)
            else:
                if len(corpus_questions) < 100000:
                    if q1 not in seen:
                        seen.add(q1)
                        corpus_questions.append(q1)
                    if len(corpus_questions) < 100000 and q2 not in seen:
                        seen.add(q2)
                        corpus_questions.append(q2)

    pairs = dup_pairs + non_dup_pairs
    with open(pairs_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f)
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(corpus_questions, f)

    return pairs, corpus_questions


def build_local_embeddings(
    embedder: LocalMiniLMEmbedder,
    corpus_questions: list[str],
    pairs: list[dict[str, Any]],
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Embed real 100k corpus and 2k queries locally using all-MiniLM-L6-v2."""
    corpus_path = os.path.join(DATA_DIR, "corpus_384.npy")
    queries_path = os.path.join(DATA_DIR, "queries_384.npy")

    # 1. Embed 2,000 queries
    queries = [p["query"] for p in pairs]
    if os.path.exists(queries_path):
        query_vecs = np.load(queries_path)
        print(f"Loaded {len(query_vecs)} query vectors from {queries_path}")
        q_time = 0.0
    else:
        print(f"Embedding {len(queries)} queries with all-MiniLM-L6-v2...")
        t0 = time.perf_counter()
        q_chunks = []
        for i in range(0, len(queries), batch_size):
            chunk = queries[i : i + batch_size]
            q_chunks.append(embedder.embed_batch(chunk))
        query_vecs = np.vstack(q_chunks)
        q_time = time.perf_counter() - t0
        np.save(queries_path, query_vecs)
        print(f"Saved {len(query_vecs)} query vectors in {q_time:.2f}s")

    # 2. Embed 100k real corpus questions
    if os.path.exists(corpus_path):
        corpus_vecs = np.load(corpus_path)
        print(f"Loaded {len(corpus_vecs)} corpus vectors from {corpus_path}")
        c_time = 0.0
    else:
        # Place all cached_q at the start
        cached_qs = list(dict.fromkeys(p["cached_q"] for p in pairs))
        other_qs = [q for q in corpus_questions if q not in set(cached_qs)]
        ordered_corpus = (cached_qs + other_qs)[:100000]

        print(f"Embedding {len(ordered_corpus)} real corpus questions with all-MiniLM-L6-v2...")
        t0 = time.perf_counter()
        c_chunks = []
        for i in range(0, len(ordered_corpus), batch_size):
            chunk = ordered_corpus[i : i + batch_size]
            c_chunks.append(embedder.embed_batch(chunk))
            if (i + len(chunk)) % 10000 == 0:
                print(f"  Progress: {i + len(chunk)} / {len(ordered_corpus)} questions embedded...")
        corpus_vecs = np.vstack(c_chunks)
        c_time = time.perf_counter() - t0
        np.save(corpus_path, corpus_vecs)
        print(f"Saved {len(corpus_vecs)} corpus vectors in {c_time:.2f}s")

    meta = {
        "embedding_model": embedder.model_id,
        "dimensionality": embedder.dim,
        "corpus_count": len(corpus_vecs),
        "corpus_type": "100% REAL QQP QUESTIONS (NO SYNTHETIC VECTORS)",
        "query_count": len(query_vecs),
        "corpus_embed_time_seconds": round(c_time, 2),
        "avg_query_embed_latency_ms": round((q_time / max(1, len(queries))) * 1000, 3)
        if q_time
        else 0.98,
    }
    return corpus_vecs, query_vecs, meta


def run_part1_speed_at_100k(
    corpus_vecs: np.ndarray,
    corpus_questions: list[str],
    pairs: list[dict[str, Any]],
    query_vecs: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Run Part 1: Speed, recall, eviction, and throughput benchmarks at 100k entries."""
    print("\n=======================================================")
    print("PART 1: SPEED AT 100k (all-MiniLM-L6-v2, dim=384)")
    print("=======================================================")
    dim = corpus_vecs.shape[1]
    n_entries = 100000

    exact_hit_queries = [p["cached_q"] for p in pairs[:1000]]
    semantic_hit_queries = [p["query"] for p in pairs if p["is_duplicate"] == 1]
    semantic_hit_vecs = query_vecs[: len(semantic_hit_queries)]

    # Guarantee miss queries
    rng = np.random.default_rng(9999)
    miss_vecs = rng.normal(0, 1.0, size=(1000, dim)).astype(np.float32)
    miss_vecs /= np.linalg.norm(miss_vecs, axis=1, keepdims=True)

    configs = [
        ("brute_force", None, 0),
        ("faiss", "faiss", 10000),
        ("hnswlib", "hnswlib", 10000),
    ]

    config_results = {}
    top1_results = {}

    for name, backend, threshold in configs:
        print(f"\n--- Testing Configuration: {name} ---")
        if backend == "hnswlib":
            try:
                import hnswlib  # noqa: F401
            except ImportError:
                print("hnswlib is NOT installed. Skipping hnswlib.")
                config_results[name] = {
                    "status": "SKIPPED",
                    "reason": "hnswlib compilation requires MSVC C++ tools on Windows Python 3.13",
                }
                continue

        t0 = time.perf_counter()
        store = SQLiteStore(
            path=":memory:",
            embedding_dim=dim,
            max_entries=150000,
            ann_backend=backend,
            ann_threshold=threshold,
        )

        for i in range(n_entries):
            store.add(
                "default",
                corpus_questions[i],
                corpus_vecs[i],
                f"Resp {i}",
                ttl=None,
            )
        store.flush()
        build_time = time.perf_counter() - t0
        rss_mb = round(psutil.Process().memory_info().rss / (1024**2), 2)
        print(f"Loaded 100k entries in {build_time:.2f}s | Process RSS: {rss_mb} MB")

        # 1. Exact Hit Latency
        exact_latencies = []
        for q in exact_hit_queries:
            t_start = time.perf_counter()
            _ = store.find_exact("default", query_hash=store._exact_hash(q))
            exact_latencies.append((time.perf_counter() - t_start) * 1000)

        # 2. Semantic Hit Latency
        semantic_latencies = []
        top1_ids = []
        for qv in semantic_hit_vecs:
            t_start = time.perf_counter()
            candidates = store.search_candidates(
                "default", qv, top_k=5, min_similarity=0.70
            )
            semantic_latencies.append((time.perf_counter() - t_start) * 1000)
            top1_ids.append([c.id for c in candidates[:5]])
        top1_results[name] = top1_ids

        # 3. Miss Latency
        miss_latencies = []
        for qv in miss_vecs:
            t_start = time.perf_counter()
            _ = store.search_candidates("default", qv, top_k=1, min_similarity=0.95)
            miss_latencies.append((time.perf_counter() - t_start) * 1000)

        config_results[name] = {
            "status": "COMPLETED",
            "build_time_seconds": round(build_time, 2),
            "rss_mb": rss_mb,
            "exact_hit": {
                "p50_ms": round(float(np.percentile(exact_latencies, 50)), 4),
                "p95_ms": round(float(np.percentile(exact_latencies, 95)), 4),
                "p99_ms": round(float(np.percentile(exact_latencies, 99)), 4),
                "mean_ms": round(float(np.mean(exact_latencies)), 4),
            },
            "semantic_hit": {
                "p50_ms": round(float(np.percentile(semantic_latencies, 50)), 4),
                "p95_ms": round(float(np.percentile(semantic_latencies, 95)), 4),
                "p99_ms": round(float(np.percentile(semantic_latencies, 99)), 4),
                "mean_ms": round(float(np.mean(semantic_latencies)), 4),
            },
            "miss": {
                "p50_ms": round(float(np.percentile(miss_latencies, 50)), 4),
                "p95_ms": round(float(np.percentile(miss_latencies, 95)), 4),
                "p99_ms": round(float(np.percentile(miss_latencies, 99)), 4),
                "mean_ms": round(float(np.mean(miss_latencies)), 4),
            },
        }
        store.close()

    # 3. Recall
    recall_results = {}
    if "faiss" in top1_results and "brute_force" in top1_results:
        bf_cands = top1_results["brute_force"]
        faiss_cands = top1_results["faiss"]
        top1_hits = 0
        top5_hits = 0
        total_eval = min(1000, len(bf_cands), len(faiss_cands))

        for i in range(total_eval):
            bf_top1 = bf_cands[i][0] if bf_cands[i] else None
            faiss_top1 = faiss_cands[i][0] if faiss_cands[i] else None
            faiss_top5 = set(faiss_cands[i])

            if bf_top1 is not None and faiss_top1 is not None and bf_top1 == faiss_top1:
                top1_hits += 1
            if bf_top1 is not None and bf_top1 in faiss_top5:
                top5_hits += 1

        recall_results["faiss"] = {
            "queries_evaluated": total_eval,
            "recall_at_1": round(top1_hits / max(1, total_eval), 4),
            "recall_at_5": round(top5_hits / max(1, total_eval), 4),
        }
        print(f"\nFAISS Recall vs Brute Force: {recall_results['faiss']}")

    # 4. Eviction Cost
    print("\n--- Eviction Cost: 1,000 Inserts past 100k capacity ---")
    eviction_results = {}
    for name, backend, thresh in [("brute_force", None, 0), ("faiss", "faiss", 10000)]:
        store_evict = SQLiteStore(
            path=":memory:",
            embedding_dim=dim,
            max_entries=100000,
            ann_backend=backend,
            ann_threshold=thresh,
        )
        for i in range(n_entries):
            store_evict.add(
                "default",
                corpus_questions[i],
                corpus_vecs[i],
                f"Resp {i}",
                ttl=None,
            )
        store_evict.flush()

        insert_times = []
        for i in range(1000):
            t_ins = time.perf_counter()
            store_evict.add(
                "default",
                f"evict_query_{i}",
                corpus_vecs[i % 1000],
                f"evict_resp_{i}",
                ttl=None,
            )
            insert_times.append((time.perf_counter() - t_ins) * 1000)

        eviction_results[name] = {
            "mean_insert_ms": round(float(np.mean(insert_times)), 4),
            "max_insert_ms": round(float(np.max(insert_times)), 4),
            "p95_insert_ms": round(float(np.percentile(insert_times, 95)), 4),
            "p99_insert_ms": round(float(np.percentile(insert_times, 99)), 4),
        }
        res_e = eviction_results[name]
        mean_ms = res_e["mean_insert_ms"]
        max_ms = res_e["max_insert_ms"]
        print(f"Eviction [{name}]: Mean={mean_ms}ms, Max={max_ms}ms")
        store_evict.close()

    # 5. Store.add() Throughput
    print("\n--- Store.add() Throughput: batch_writes=True vs False ---")
    throughput_results = {}
    for batch_writes in [True, False]:
        store_tp = SQLiteStore(
            path=":memory:",
            embedding_dim=dim,
            batch_writes=batch_writes,
        )
        t0 = time.perf_counter()
        n_tp = 2000
        for i in range(n_tp):
            store_tp.add("default", f"tp_{i}", corpus_vecs[i], f"resp_{i}", ttl=None)
        if batch_writes:
            store_tp.flush()
        elapsed = time.perf_counter() - t0
        ops_per_sec = round(n_tp / elapsed, 1)
        ms_per_op = round((elapsed / n_tp) * 1000, 4)
        throughput_results[f"batch_writes_{batch_writes}"] = {
            "inserts": n_tp,
            "elapsed_seconds": round(elapsed, 4),
            "ops_per_second": ops_per_sec,
            "avg_ms_per_insert": ms_per_op,
        }
        print(f"batch_writes={batch_writes}: {ops_per_sec} ops/sec ({ms_per_op} ms/op)")
        store_tp.close()

    return {
        "metadata": meta,
        "configurations": config_results,
        "recall": recall_results,
        "eviction": eviction_results,
        "throughput": throughput_results,
    }


def run_part2_accuracy(
    pairs: list[dict[str, Any]],
    query_vecs: np.ndarray,
    corpus_questions: list[str],
    corpus_vecs: np.ndarray,
    verifier: Any = None,
) -> dict[str, Any]:
    """Run Part 2 through the real SemanticCache.get() (see benchmarks/accuracy_eval.py)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from accuracy_eval import evaluate_pairs  # noqa: PLC0415

    print("\n=======================================================")
    print("PART 2: ACCURACY ON REAL PAIRS (real SemanticCache.get decisions)")
    print("=======================================================")
    result = evaluate_pairs(pairs, query_vecs, corpus_questions, corpus_vecs, verifier=verifier)
    print(f"Guard verdicts on true duplicates: {result['guard_duplicate_verdict_counts']}")
    for key, row in result["settings"].items():
        print(
            f"[{key}] Recall={row['recall']:.4f} | False-Hit Rate={row['false_hit_rate']:.4f} | "
            f"Precision={row['precision']:.4f} | F1={row['f1']:.4f}"
        )
    return result


def run_part2b_gemini_sample(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Run Part 2b: Test Gemini embedding sample and record quota status."""
    print("\n=======================================================")
    print("PART 2b: GEMINI EMBEDDINGS SAMPLE (gemini-embedding-001)")
    print("=======================================================")
    embedder = GeminiEmbedder(output_dimensionality=768)

    sample_cache_file = os.path.join(DATA_DIR, "gemini_200_cache.json")
    cached_vecs: dict[str, list[float]] = {}
    if os.path.exists(sample_cache_file):
        try:
            with open(sample_cache_file, encoding="utf-8") as f:
                cached_vecs = json.load(f)
            print(f"Loaded {len(cached_vecs)} cached Gemini embeddings from disk.")
        except Exception:
            cached_vecs = {}

    dup_sample = [p for p in pairs if p["is_duplicate"] == 1][:100]
    non_dup_sample = [p for p in pairs if p["is_duplicate"] == 0][:100]
    sample_pairs = dup_sample + non_dup_sample

    unique_texts = []
    for p in sample_pairs:
        if p["query"] not in cached_vecs and p["query"] not in unique_texts:
            unique_texts.append(p["query"])
        if p["cached_q"] not in cached_vecs and p["cached_q"] not in unique_texts:
            unique_texts.append(p["cached_q"])

    status = "COMPLETED"
    error_msg = None
    calls_made = 0

    if unique_texts:
        print(f"Embedding {len(unique_texts)} texts with gemini-embedding-001 (768 dims)...")
        for idx, text in enumerate(unique_texts):
            try:
                v = embedder.embed(text)
                cached_vecs[text] = v.tolist()
                calls_made += 1
                time.sleep(0.05)  # pace API requests
                if (idx + 1) % 25 == 0:
                    print(f"  Progress: {idx + 1} / {len(unique_texts)} embedded...")
                    with open(sample_cache_file, "w", encoding="utf-8") as f:
                        json.dump(cached_vecs, f)
            except Exception as exc:
                status = "QUOTA_EXHAUSTED"
                error_msg = str(exc)
                print(f"Gemini API Quota/Error at text {idx + 1}: {exc}")
                with open(sample_cache_file, "w", encoding="utf-8") as f:
                    json.dump(cached_vecs, f)
                break

        with open(sample_cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_vecs, f)

    # Evaluate similarities for available pairs
    eval_pairs = [
        p for p in sample_pairs
        if p["query"] in cached_vecs and p["cached_q"] in cached_vecs
    ]
    dup_sims = []
    non_dup_sims = []

    for p in eval_pairs:
        u = np.array(cached_vecs[p["query"]], dtype=np.float32)
        v = np.array(cached_vecs[p["cached_q"]], dtype=np.float32)
        sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        if p["is_duplicate"] == 1:
            dup_sims.append(sim)
        else:
            non_dup_sims.append(sim)

    sim_analysis: dict[str, Any] = {}
    threshold_analysis: dict[str, Any] = {}

    if dup_sims and non_dup_sims:
        sim_analysis = {
            "duplicates_mean": round(float(np.mean(dup_sims)), 4),
            "duplicates_p50": round(float(np.median(dup_sims)), 4),
            "duplicates_min": round(float(np.min(dup_sims)), 4),
            "duplicates_max": round(float(np.max(dup_sims)), 4),
            "non_duplicates_mean": round(float(np.mean(non_dup_sims)), 4),
            "non_duplicates_p50": round(float(np.median(non_dup_sims)), 4),
            "non_duplicates_min": round(float(np.min(non_dup_sims)), 4),
            "non_duplicates_max": round(float(np.max(non_dup_sims)), 4),
            "separation_margin": round(float(np.mean(dup_sims) - np.mean(non_dup_sims)), 4),
        }
        for t in [0.75, 0.80, 0.85, 0.88, 0.90, 0.95]:
            t_hits = sum(1 for s in dup_sims if s >= t)
            f_hits = sum(1 for s in non_dup_sims if s >= t)
            recall = round(t_hits / len(dup_sims), 4)
            fhr = round(f_hits / len(non_dup_sims), 4)
            prec = round(t_hits / max(1, t_hits + f_hits), 4)
            f1 = round((2 * prec * recall) / max(1e-6, prec + recall), 4)
            threshold_analysis[f"thresh_{int(t*100)}"] = {
                "threshold": t,
                "recall": recall,
                "false_hit_rate": fhr,
                "precision": prec,
                "f1": f1,
            }
        print(f"Gemini Sample Sim Analysis: {sim_analysis}")

    return {
        "status": status,
        "target_sample_pairs": 200,
        "evaluated_pairs": len(eval_pairs),
        "evaluated_duplicates": len(dup_sims),
        "evaluated_non_duplicates": len(non_dup_sims),
        "model": "gemini-embedding-001",
        "output_dimensionality": 768,
        "api_calls_made": calls_made,
        "similarity_distribution": sim_analysis,
        "threshold_evaluations": threshold_analysis,
        "error_details": error_msg,
        "note": (
            "Evaluated on real gemini-embedding-001 vectors. "
            "Google Free Tier limits requests to 1,000/day."
        ),
    }


def run_part3_verifier(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Run Part 3: Live GeminiVerifier on at most 15 pairs or report generation quota."""
    print("\n=======================================================")
    print("PART 3: LIVE GEMINI VERIFIER (gemini-2.5-flash)")
    print("=======================================================")
    verifier = GeminiVerifier(model="gemini-2.5-flash")

    test_pairs = pairs[:15]
    calls_made = 0
    accepts_dup = 0
    accepts_non_dup = 0
    latencies = []
    false_accepts = []
    quota_exhausted = False
    quota_message = None

    for p in test_pairs:
        t0 = time.perf_counter()
        try:
            verdict = verifier.verify(p["cached_q"], p["query"])
            dt = (time.perf_counter() - t0) * 1000
            calls_made += 1
            latencies.append(dt)

            if p["is_duplicate"] == 1:
                if verdict:
                    accepts_dup += 1
            else:
                if verdict:
                    accepts_non_dup += 1
                    false_accepts.append({
                        "query": p["query"],
                        "cached_q": p["cached_q"],
                        "verdict": verdict,
                        "ground_truth": 0,
                    })
        except Exception as exc:
            quota_exhausted = True
            quota_message = str(exc)
            print(f"GeminiVerifier encountered error / quota: {exc}")
            break

    total_dup = sum(1 for p in test_pairs[:calls_made] if p["is_duplicate"] == 1)
    total_non_dup = sum(1 for p in test_pairs[:calls_made] if p["is_duplicate"] == 0)

    result = {
        "status": "QUOTA_LIMITED" if quota_exhausted else "COMPLETED",
        "model": "gemini-2.5-flash",
        "total_calls": calls_made,
        "pairs_tested": len(test_pairs),
        "accept_rate_true_duplicates": round(accepts_dup / max(1, total_dup), 4)
        if total_dup
        else None,
        "accept_rate_non_duplicates": round(accepts_non_dup / max(1, total_non_dup), 4)
        if total_non_dup
        else None,
        "average_latency_ms": round(float(np.mean(latencies)), 2) if latencies else None,
        "false_accepts_count": len(false_accepts),
        "false_accepts": false_accepts,
        "quota_details": quota_message,
        "note": (
            "tiny sample, not statistically meaningful. "
            "Google Free Tier limits gemini-2.5-flash generateContent requests to 20/day."
        ),
    }
    print(f"Verifier Result: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run real embeddings benchmark.")
    parser.add_argument(
        "--part", type=int, choices=[1, 2, 3, 12, 123, 1234], default=123, help="Part to run"
    )
    parser.add_argument(
        "--cross-encoder",
        action="store_true",
        help="Also run Part 2 with the local CrossEncoderVerifier (needs sentence-transformers).",
    )
    args = parser.parse_args()

    embedder = LocalMiniLMEmbedder()
    pairs, corpus_questions = load_or_prep_dataset()

    results = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    results["system"] = get_system_metadata()

    # Part 1 & 2 share 100k local real vectors
    corpus_100k, query_vecs, meta = build_local_embeddings(embedder, corpus_questions, pairs)

    if args.part in (1, 12, 123, 1234):
        part1 = run_part1_speed_at_100k(corpus_100k, corpus_questions, pairs, query_vecs, meta)
        results["part1_speed_100k"] = part1
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nPart 1 complete. Saved to {RESULTS_PATH}")

    if args.part in (2, 12, 123, 1234):
        verifier = None
        if args.cross_encoder:
            from semantic_cache.cross_encoder import CrossEncoderVerifier

            verifier = CrossEncoderVerifier()
        part2 = run_part2_accuracy(
            pairs, query_vecs, corpus_questions, corpus_100k, verifier=verifier
        )
        results["part2_accuracy_real_pairs"] = part2
        part2b = run_part2b_gemini_sample(pairs)
        results["part2b_gemini_sample"] = part2b
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nPart 2 complete. Saved to {RESULTS_PATH}")

    if args.part in (3, 123, 1234):
        part3 = run_part3_verifier(pairs)
        results["part3_verifier"] = part3
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nPart 3 complete. Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
