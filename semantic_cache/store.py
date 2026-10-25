"""SQLite-backed storage with an in-memory vector index for fast cosine search.

Security notes:
  - All SQL uses parameterised queries — no string interpolation.
  - Namespace values are validated; they may appear in filenames or log output.
  - SQLite files are NOT encrypted at rest.  Apply OS-level permissions to the
    database file.  Do not store the file in a world-readable location.
  - Backups of the SQLite file contain all cached responses.  Treat them with
    the same sensitivity as the original data.
  - This store does NOT implement per-tenant authorisation.  Namespace isolation
    is a *partitioning* mechanism, not a security boundary.

Concurrency:
  - A single connection is opened with ``check_same_thread=False``.
  - All mutation paths acquire ``_lock`` (threading.Lock).
  - This provides safe concurrent access within one process.
  - Cross-process safety is NOT guaranteed (SQLite WAL mode recommended for
    multi-reader, single-writer cross-process access).
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import sqlite3
import threading
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .embedders import validate_embedding
from .guard import exact_match_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
_MAX_NAMESPACE_LEN = 256
_MAX_QUERY_LEN = 32_768
_MAX_RESPONSE_LEN = 1_048_576  # 1 MB
_MAX_TTL = 60 * 60 * 24 * 365  # 1 year in seconds


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    id: int
    query: str
    response: str
    score: float = 0.0
    expires_at: float | None = None
    hits: int = 0
    last_hit_at: float | None = None
    created_at: float = 0.0
    namespace: str = ""
    embedder: str = ""
    query_hash: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_namespace(ns: str) -> str:
    if not isinstance(ns, str) or not ns:
        raise ValueError("namespace must be a non-empty string")
    if len(ns) > _MAX_NAMESPACE_LEN:
        raise ValueError(f"namespace exceeds {_MAX_NAMESPACE_LEN} characters: {ns[:40]!r}...")
    return ns


def _validate_ttl(ttl: float | None) -> None:
    if ttl is not None:
        if not isinstance(ttl, (int, float)) or ttl <= 0:
            raise ValueError(f"ttl must be a positive number, got {ttl!r}")
        if ttl > _MAX_TTL:
            raise ValueError(f"ttl exceeds maximum of {_MAX_TTL} seconds")


def _safe_load_blob(blob: bytes, expected_dim: int) -> np.ndarray | None:
    """Deserialise a stored embedding BLOB.  Returns None on corruption."""
    try:
        # np.frombuffer returns read-only view — copy immediately
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        return validate_embedding(vec, expected_dim=expected_dim)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping corrupt embedding BLOB: %s", exc)
        return None


def _flush_store_at_exit(ref: weakref.ReferenceType) -> None:
    """Flush queued writes on normal interpreter exit (stores are often never closed)."""
    store = ref()
    if store is not None:
        try:
            store.flush()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# VectorIndex with optional ANN backend (FAISS / hnswlib)
# ---------------------------------------------------------------------------


def _build_ann_index(dim: int, matrix: np.ndarray, backend: str | None = None) -> Any:
    """Instantiate and populate an ANN index if an engine is requested."""
    if not backend:
        return None
    if backend in ("auto", "faiss"):
        try:
            import faiss

            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
            if matrix.shape[0] > 0:
                index.add(np.ascontiguousarray(matrix, dtype=np.float32))
            return index
        except ImportError:
            if backend == "faiss":
                raise
    if backend in ("auto", "hnswlib"):
        try:
            import hnswlib

            index = hnswlib.Index(space="cosine", dim=dim)
            count = matrix.shape[0]
            index.init_index(max_elements=max(1000, count * 2), ef_construction=100, M=16)
            index.set_ef(64)
            if count > 0:
                index.add_items(matrix, np.arange(count))
            return index
        except ImportError:
            if backend == "hnswlib":
                raise
    return None


class VectorIndex:
    """In-memory vector index with capacity-doubling growth and optional ANN backend.

    Pre-allocates an initial matrix capacity and doubles it whenever the
    capacity is reached. Above `ann_threshold`, switches to an ANN backend
    (FAISS / hnswlib) if available; uses brute-force matrix multiplication otherwise.
    """

    __slots__ = (
        "ids",
        "matrix",
        "count",
        "dim",
        "ann_threshold",
        "ann_backend",
        "ann_index",
    )

    def __init__(
        self,
        ids: list[int],
        matrix: np.ndarray,
        dim: int,
        ann_threshold: int = 10_000,
        ann_backend: str | None = None,
    ) -> None:
        self.ids = list(ids)
        self.count = len(self.ids)
        self.dim = dim
        self.ann_threshold = ann_threshold
        self.ann_backend = ann_backend
        self.ann_index: Any = None
        capacity = max(16, self.count)
        if matrix.shape[0] < capacity:
            self.matrix = np.zeros((capacity, dim), dtype=np.float32)
            if self.count > 0:
                self.matrix[: self.count] = matrix[: self.count]
        else:
            self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)

        if self.count >= self.ann_threshold:
            self.ann_index = _build_ann_index(self.dim, self.active_matrix(), self.ann_backend)

    def __iter__(self) -> Iterator[Any]:
        """Allows unpacking as `ids, mat = index` for backward compatibility."""
        return iter((self.ids, self.active_matrix()))

    def __getitem__(self, item: int) -> Any:
        if item == 0:
            return self.ids
        elif item == 1:
            return self.active_matrix()
        raise IndexError(f"VectorIndex index out of range: {item}")

    def append(self, entry_id: int, vec: np.ndarray) -> None:
        flat_vec = vec.reshape(-1)
        dim = flat_vec.shape[0]
        if self.dim == 0 and dim > 0:
            self.dim = dim
            if self.matrix.shape[1] == 0:
                self.matrix = np.zeros((max(16, self.count + 1), dim), dtype=np.float32)

        if self.count >= self.matrix.shape[0]:
            new_capacity = max(16, self.matrix.shape[0] * 2)
            new_mat = np.zeros((new_capacity, self.dim), dtype=np.float32)
            if self.count > 0:
                new_mat[: self.count] = self.matrix[: self.count]
            self.matrix = new_mat

        self.matrix[self.count] = flat_vec
        self.ids.append(entry_id)
        self.count += 1

        if self.count >= self.ann_threshold:
            if self.ann_index is None:
                self.ann_index = _build_ann_index(self.dim, self.active_matrix(), self.ann_backend)
            else:
                try:
                    vec32 = flat_vec.reshape(1, -1).astype(np.float32)
                    if hasattr(self.ann_index, "add_items"):  # hnswlib
                        if self.count > self.ann_index.get_max_elements():
                            self.ann_index.resize_index(self.count * 2)
                        self.ann_index.add_items(vec32, np.array([self.count - 1]))
                    elif hasattr(self.ann_index, "add"):  # faiss
                        self.ann_index.add(vec32)
                    else:
                        self.ann_index = None
                except Exception:
                    self.ann_index = None

    def active_matrix(self) -> np.ndarray:
        return self.matrix[: self.count]

    def remove_ids(self, ids_to_remove: set[int]) -> None:
        """In-place removal of evicted IDs with array compaction."""
        if not ids_to_remove or self.count == 0:
            return
        keep_indices = [i for i, eid in enumerate(self.ids) if eid not in ids_to_remove]
        if len(keep_indices) == self.count:
            return
        if not keep_indices:
            self.ids.clear()
            self.count = 0
            self.ann_index = None
            return
        # Compact matrix rows in-place
        self.matrix[: len(keep_indices)] = self.matrix[keep_indices]
        self.ids = [self.ids[i] for i in keep_indices]
        self.count = len(self.ids)
        if self.ann_index is not None:
            if self.count >= self.ann_threshold:
                self.ann_index = _build_ann_index(self.dim, self.active_matrix(), self.ann_backend)
            else:
                self.ann_index = None

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        min_similarity: float = -1.0,
    ) -> tuple[list[int], list[float]]:
        """Search nearest vectors using ANN backend if enabled/built, otherwise brute force."""
        if not self.ids or self.count == 0:
            return [], []

        if self.ann_index is not None and self.count >= self.ann_threshold:
            try:
                q = query_vec.reshape(1, -1).astype(np.float32)
                if hasattr(self.ann_index, "knn_query"):  # hnswlib (cosine distance)
                    labels, dists = self.ann_index.knn_query(q, k=min(top_k, self.count))
                    indices, scores = labels, 1.0 - dists
                else:  # faiss (inner product)
                    scores, indices = self.ann_index.search(q, top_k)
                res_ids: list[int] = []
                res_scores: list[float] = []
                for idx, score in zip(indices[0], scores[0]):
                    if 0 <= idx < len(self.ids):
                        s = float(score)
                        if s >= min_similarity:
                            res_ids.append(self.ids[int(idx)])
                            res_scores.append(s)
                return res_ids, res_scores
            except Exception as exc:
                logger.debug("ANN search failed, falling back to brute force: %s", exc)

        # Brute force cosine / dot product
        scores = self.active_matrix() @ query_vec
        eligible = np.flatnonzero(scores >= float(min_similarity))
        if eligible.size == 0:
            return [], []
        order = eligible[np.argsort(scores[eligible])[::-1][:top_k]]
        return [self.ids[int(i)] for i in order], [float(scores[int(i)]) for i in order]


class _DBProxy:
    """Proxy around sqlite3.Connection that flushes pending writes before execution."""

    def __init__(self, conn: sqlite3.Connection, store: SQLiteStore) -> None:
        self._conn = conn
        self._store = store

    def execute(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        self._store.flush()
        return self._conn.execute(*args, **kwargs)

    def executemany(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        self._store.flush()
        return self._conn.executemany(*args, **kwargs)

    def commit(self) -> None:
        self._store.flush()
        return self._conn.commit()

    def rollback(self) -> None:
        return self._conn.rollback()

    def close(self) -> None:
        self._store.flush()
        return self._conn.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------


class SQLiteStore:
    """SQLite-backed semantic cache storage.

    Parameters
    ----------
    path:
        Path to the SQLite file.  Defaults to ``:memory:`` (non-persistent).
        Use a file path for persistence across restarts.
    embedding_dim:
        Expected embedding dimension.  If set, corrupt BLOBs with the wrong
        dimension are skipped during index loading rather than causing a crash.
        Set to ``None`` to skip dimension validation (not recommended).
    max_entries:
        If set, evict the least-recently-used entries when the total entry
        count reaches this limit.  Applied per-write, so the store may
        momentarily exceed this by 1.
    """

    def __init__(
        self,
        path: str = ":memory:",
        embedding_dim: int | None = None,
        max_entries: int | None = None,
        batch_writes: bool = True,
        ann_threshold: int = 10_000,
        ann_backend: str | None = None,
        eviction_policy: str = "lru",
    ) -> None:
        if embedding_dim is not None and embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer or None")
        if max_entries is not None and (not isinstance(max_entries, int) or max_entries < 1):
            raise ValueError("max_entries must be a positive integer or None")
        self._path = path
        self._embedding_dim = embedding_dim
        self._max_entries = max_entries
        self._batch_writes = batch_writes
        self._ann_threshold = ann_threshold
        self._ann_backend = ann_backend
        self._eviction_policy = eviction_policy.lower()
        if self._eviction_policy not in ("lru", "lfu", "fifo"):
            raise ValueError(f"Invalid eviction_policy: {eviction_policy!r}")
        self._lock = threading.RLock()
        self._flushing = False
        self._uncommitted_writes = 0
        self._pending_hits: dict[int, tuple[int, float]] = {}

        # A generous busy timeout lets concurrent processes queue for write locks
        # (for example two app instances starting against the same cache file).
        raw_conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._raw_db = raw_conn
        self.db: Any = _DBProxy(raw_conn, self)
        try:
            self._enable_wal()
            self._raw_db.execute("PRAGMA foreign_keys=ON")
            self._raw_db.execute(
                """CREATE TABLE IF NOT EXISTS entries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace   TEXT    NOT NULL,
                    query       TEXT    NOT NULL,
                    embedding   BLOB    NOT NULL,
                    response    TEXT    NOT NULL,
                    created_at  REAL    NOT NULL,
                    expires_at  REAL,
                    hits        INTEGER NOT NULL DEFAULT 0,
                    last_hit_at REAL,
                    query_hash  TEXT,
                    embedder    TEXT    NOT NULL DEFAULT ''
                )"""
            )

            # Concurrency-safe migration: use user_version as a migration lock.
            # 0 = not migrated, 1 = migration in progress, 2 = migration complete.
            self._run_safe_migration()

            self._raw_db.execute("CREATE INDEX IF NOT EXISTS ix_ns ON entries(namespace)")
            self._raw_db.execute(
                "CREATE INDEX IF NOT EXISTS ix_ns_exp ON entries(namespace, expires_at)"
            )
            self._raw_db.execute(
                "CREATE INDEX IF NOT EXISTS ix_ns_hash ON entries(namespace, query_hash)"
            )
            self._raw_db.execute(
                "CREATE INDEX IF NOT EXISTS ix_ns_emb ON entries(namespace, embedder)"
            )
            self._raw_db.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_exact "
                "ON entries(namespace, query_hash, embedder)"
            )
            self._raw_db.commit()
        except BaseException:
            try:
                self._raw_db.close()
            except Exception:
                pass
            raise

        # In-memory index structures for zero-disk lookups
        self._entries: dict[int, Entry] = {}
        self._exact_index: dict[tuple[str, str, str], int] = {}
        self._entries_loaded: set[str] = set()
        self._index: dict[str, VectorIndex] = {}

        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        if self._batch_writes:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="semantic-cache-db-writer",
                daemon=True,
            )
            self._writer_thread.start()
            atexit.register(_flush_store_at_exit, weakref.ref(self))

    def _writer_loop(self) -> None:
        while not self._stop_event.wait(0.05):
            try:
                self.flush()
            except Exception:
                logger.debug("Error in background writer loop", exc_info=True)

    def _flush_unlocked(self) -> None:
        if self._flushing or getattr(self, "_raw_db", None) is None:
            return
        self._flushing = True
        try:
            hits_to_flush = []
            if self._pending_hits:
                hits_to_flush = list(self._pending_hits.items())
                self._pending_hits.clear()
                if self._raw_db is not None:
                    self._raw_db.executemany(
                        "UPDATE entries SET hits = hits + ?, last_hit_at = ? WHERE id = ?",
                        [(count, ts, eid) for eid, (count, ts) in hits_to_flush],
                    )
            if (self._uncommitted_writes > 0 or hits_to_flush) and self._raw_db is not None:
                self._raw_db.commit()
                self._uncommitted_writes = 0
        finally:
            self._flushing = False

    def flush(self) -> None:
        """Flush pending queued inserts and hit updates to SQLite."""
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                self._flush_unlocked()
        else:
            self._flush_unlocked()

    def _enable_wal(self) -> None:
        """Switch the database to WAL mode without racing other processes."""
        if self._path == ":memory:":
            return
        for attempt in range(40):
            try:
                mode = self._raw_db.execute("PRAGMA journal_mode").fetchone()[0]
                if str(mode).lower() != "wal":
                    self._raw_db.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                time.sleep(0.05 * (1 + attempt % 5))
        logger.warning("Could not enable WAL mode (database busy); using the default journal mode")

    def _run_safe_migration(self) -> None:
        """Run schema migration with transactional concurrency lock via BEGIN IMMEDIATE."""
        import time as _time

        version = self._raw_db.execute("PRAGMA user_version").fetchone()[0]
        if version >= 2:
            return

        max_attempts = 40
        poll_interval = 0.05

        for attempt in range(max_attempts):
            try:
                self._raw_db.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    _time.sleep(poll_interval)
                    v = self._raw_db.execute("PRAGMA user_version").fetchone()[0]
                    if v >= 2:
                        return
                    continue
                raise
        else:
            raise sqlite3.OperationalError(
                f"Could not acquire migration lock after {max_attempts} attempts"
            )

        try:
            version = self._raw_db.execute("PRAGMA user_version").fetchone()[0]
            if version >= 2:
                self._raw_db.execute("COMMIT")
                return

            existing_cols = {
                row[1] for row in self._raw_db.execute("PRAGMA table_info(entries)").fetchall()
            }
            if "query_hash" not in existing_cols:
                self._raw_db.execute("ALTER TABLE entries ADD COLUMN query_hash TEXT")
            if "embedder" not in existing_cols:
                self._raw_db.execute(
                    "ALTER TABLE entries ADD COLUMN embedder TEXT NOT NULL DEFAULT ''"
                )

            unhashed = self._raw_db.execute(
                "SELECT id, query FROM entries WHERE query_hash IS NULL OR query_hash = ''"
            ).fetchall()
            if unhashed:
                updates = [
                    (hashlib.sha256(exact_match_key(q).encode("utf-8")).hexdigest(), rid)
                    for rid, q in unhashed
                ]
                self._raw_db.executemany("UPDATE entries SET query_hash = ? WHERE id = ?", updates)

            self._raw_db.execute("PRAGMA user_version = 2")
            self._raw_db.execute("COMMIT")
        except BaseException:
            try:
                self._raw_db.execute("ROLLBACK")
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_hash(query: str) -> str:
        return hashlib.sha256(exact_match_key(query).encode("utf-8")).hexdigest()

    def add(
        self,
        namespace: str,
        query: str,
        embedding: np.ndarray,
        response: str,
        ttl: float | None,
        query_hash: str | None = None,
        embedder: str = "",
    ) -> int:
        """Insert a new cache entry. Returns the new entry ID."""
        _validate_namespace(namespace)
        _validate_ttl(ttl)
        embedding = validate_embedding(embedding, self._embedding_dim)

        if len(query.encode("utf-8")) > _MAX_QUERY_LEN:
            raise ValueError("query exceeds maximum allowed size")
        if len(response.encode("utf-8")) > _MAX_RESPONSE_LEN:
            raise ValueError("response exceeds maximum allowed size")

        now = time.time()
        blob = embedding.astype(np.float32).tobytes()
        if query_hash is None:
            query_hash = hashlib.sha256(exact_match_key(query).encode("utf-8")).hexdigest()

        with self._lock:
            index_keys = [namespace]
            if embedder:
                index_keys.append(f"{namespace}::{embedder}")

            for ikey in index_keys:
                if ikey not in self._index:
                    self._load(namespace, embedder if ikey != namespace else None)

            expires_at = now + ttl if ttl else None
            cur = self._raw_db.execute(
                """INSERT INTO entries
                   (namespace, query, embedding, response, created_at,
                    expires_at, query_hash, embedder)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    namespace,
                    query,
                    blob,
                    response,
                    now,
                    expires_at,
                    query_hash,
                    embedder,
                ),
            )
            entry_id: int = cur.lastrowid
            self._uncommitted_writes += 1
            if not self._batch_writes:
                self._raw_db.commit()
                self._uncommitted_writes = 0

            entry = Entry(
                id=entry_id,
                query=query,
                response=response,
                score=0.0,
                expires_at=expires_at,
                hits=0,
                last_hit_at=None,
                created_at=now,
                namespace=namespace,
                embedder=embedder,
                query_hash=query_hash,
            )
            self._entries[entry_id] = entry
            self._exact_index[(namespace, query_hash, embedder)] = entry_id

            for ikey in index_keys:
                if ikey in self._index:
                    self._index[ikey].append(entry_id, embedding)

            if self._max_entries is not None:
                self._evict_if_needed(protect_id=entry_id)

        logger.debug("cache.add namespace=%r id=%d", namespace, entry_id)
        return entry_id

    def _evict_if_needed(self, protect_id: int | None = None) -> None:
        """Batch-evict entries according to eviction policy down to 90% capacity."""
        self.flush()
        total = self._raw_db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if self._max_entries is not None and total > self._max_entries:
            target_count = max(0, min(self._max_entries - 1, int(self._max_entries * 0.90)))
            entries_to_evict = total - target_count

            if self._eviction_policy == "lfu":
                order_by = "hits ASC, created_at ASC"
            elif self._eviction_policy == "fifo":
                order_by = "created_at ASC"
            else:  # lru
                order_by = "COALESCE(last_hit_at, created_at) ASC"

            # Never evict the entry that was just inserted, otherwise LFU (0 hits)
            # would drop every new entry immediately once the others have hits.
            cur = self._raw_db.execute(
                f"""SELECT id FROM entries
                   WHERE id != ?
                   ORDER BY {order_by}
                   LIMIT ?""",
                (-1 if protect_id is None else protect_id, entries_to_evict),
            )
            evicted_ids = [row[0] for row in cur.fetchall()]
            if evicted_ids:
                evicted_set = set(evicted_ids)
                for eid in evicted_ids:
                    e = self._entries.pop(eid, None)
                    if e:
                        self._exact_index.pop((e.namespace, e.query_hash, e.embedder), None)
                for vindex in self._index.values():
                    vindex.remove_ids(evicted_set)

                chunk_size = 500
                for i in range(0, len(evicted_ids), chunk_size):
                    chunk = evicted_ids[i : i + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    self._raw_db.execute(
                        f"DELETE FROM entries WHERE id IN ({placeholders})",
                        chunk,
                    )
                self._raw_db.commit()

    _evict_lru_if_needed = _evict_if_needed

    # ------------------------------------------------------------------
    # Exact Match Lookup
    # ------------------------------------------------------------------

    def find_exact(
        self,
        namespace: str,
        query_hash: str,
        embedder: str | None = None,
    ) -> Entry | None:
        """Find an entry by exact query hash without embedding comparison."""
        _validate_namespace(namespace)
        now = time.time()
        with self._lock:
            index_key = f"{namespace}::{embedder}" if embedder else namespace
            if index_key not in self._entries_loaded:
                self._load(namespace, embedder)

            eid = None
            if embedder:
                eid = self._exact_index.get((namespace, query_hash, embedder))
            else:
                eid = self._exact_index.get((namespace, query_hash, ""))
                if eid is None:
                    for (ns, qh, _), cand_id in self._exact_index.items():
                        if ns == namespace and qh == query_hash:
                            eid = cand_id
                            break
            if eid is None:
                return None
            entry = self._entries.get(eid)
            if entry is None:
                return None
            if entry.expires_at is not None and entry.expires_at <= now:
                return None
            return Entry(id=entry.id, query=entry.query, response=entry.response, score=1.0)

    # ------------------------------------------------------------------
    # Index loading (internal)
    # ------------------------------------------------------------------

    def _load(self, namespace: str, embedder: str | None = None) -> VectorIndex:
        """Load (or return cached) the in-memory vector index for *namespace*.
        Caller must hold ``self._lock``.
        """
        index_key = f"{namespace}::{embedder}" if embedder else namespace
        if index_key in self._index and index_key in self._entries_loaded:
            return self._index[index_key]

        self._flush_unlocked()
        now = time.time()
        sql_cols = (
            "id, query, response, embedding, expires_at, created_at, hits, "
            "last_hit_at, query_hash, embedder"
        )
        if embedder:
            rows = self._raw_db.execute(
                f"""SELECT {sql_cols}
                   FROM entries
                   WHERE namespace=? AND embedder=?
                     AND (expires_at IS NULL OR expires_at > ?)""",
                (namespace, embedder, now),
            ).fetchall()
        else:
            rows = self._raw_db.execute(
                f"""SELECT {sql_cols}
                   FROM entries
                   WHERE namespace=? AND (expires_at IS NULL OR expires_at > ?)""",
                (namespace, now),
            ).fetchall()

        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for r in rows:
            eid, q, resp, blob, exp, cr, hits, lha, qh, emb = r
            vec = _safe_load_blob(blob, self._embedding_dim)
            if vec is not None:
                ids.append(eid)
                vecs.append(vec)
                entry = Entry(
                    id=eid,
                    query=q,
                    response=resp,
                    score=0.0,
                    expires_at=exp,
                    hits=hits,
                    last_hit_at=lha,
                    created_at=cr,
                    namespace=namespace,
                    embedder=emb,
                    query_hash=qh or "",
                )
                self._entries[eid] = entry
                if qh:
                    self._exact_index[(namespace, qh, emb)] = eid

        if vecs:
            initial_mat = np.vstack(vecs)
            dim = initial_mat.shape[1]
        else:
            dim = self._embedding_dim or 0
            initial_mat = np.zeros((0, dim), dtype=np.float32)

        vindex = VectorIndex(
            ids,
            initial_mat,
            dim,
            ann_threshold=self._ann_threshold,
            ann_backend=self._ann_backend,
        )
        self._index[index_key] = vindex
        self._entries_loaded.add(index_key)
        return vindex

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search_candidates(
        self,
        namespace: str,
        embedding: np.ndarray,
        embedder: str | None = None,
        *,
        top_k: int = 5,
        min_similarity: float = -1.0,
    ) -> list[Entry]:
        """Return up to ``top_k`` valid candidates sorted by cosine score descending."""
        _validate_namespace(namespace)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if top_k > 500:
            raise ValueError("top_k must not exceed 500")
        if not isinstance(min_similarity, (int, float)) or not np.isfinite(min_similarity):
            raise ValueError("min_similarity must be a finite number")
        if min_similarity < -1.0 or min_similarity > 1.0:
            raise ValueError("min_similarity must be in [-1.0, 1.0]")

        embedding = validate_embedding(embedding, self._embedding_dim)

        with self._lock:
            vindex = self._load(namespace, embedder)
            candidate_ids, scores = vindex.search(
                embedding, top_k=top_k, min_similarity=float(min_similarity)
            )
            if not candidate_ids:
                return []

            now = time.time()
            results: list[Entry] = []
            for eid, score in zip(candidate_ids, scores):
                entry = self._entries.get(eid)
                if entry is None:
                    continue
                if entry.expires_at is not None and entry.expires_at <= now:
                    continue
                results.append(
                    Entry(
                        id=entry.id,
                        query=entry.query,
                        response=entry.response,
                        score=float(score),
                    )
                )
            return results

    def search(
        self,
        namespace: str,
        embedding: np.ndarray,
        embedder: str | None = None,
    ) -> Entry | None:
        """Find the nearest stored entry by cosine similarity (legacy Top-1 API)."""
        candidates = self.search_candidates(
            namespace, embedding, embedder=embedder, top_k=1, min_similarity=-1.0
        )
        return candidates[0] if candidates else None

    def record_hit(self, entry_id: int) -> None:
        now = time.time()
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is not None:
                entry.hits += 1
                entry.last_hit_at = now
            if self._batch_writes:
                prev = self._pending_hits.get(entry_id, (0, now))[0]
                self._pending_hits[entry_id] = (prev + 1, now)
            else:
                self._raw_db.execute(
                    "UPDATE entries SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
                    (now, entry_id),
                )
                self._raw_db.commit()

    # ------------------------------------------------------------------
    # Deletion / maintenance
    # ------------------------------------------------------------------

    def delete_by_query(self, namespace: str, query: str) -> int:
        """Delete all entries whose stored query exactly matches *query*.
        Returns the number of deleted rows.
        """
        _validate_namespace(namespace)
        with self._lock:
            self.flush()
            cur = self._raw_db.execute(
                "DELETE FROM entries WHERE namespace=? AND query=?",
                (namespace, query),
            )
            self._raw_db.commit()
            deleted = cur.rowcount
            to_remove = [
                eid
                for eid, e in self._entries.items()
                if e.namespace == namespace and e.query == query
            ]
            for eid in to_remove:
                e = self._entries.pop(eid, None)
                if e:
                    self._exact_index.pop((e.namespace, e.query_hash, e.embedder), None)
            rem_set = set(to_remove)
            for k in list(self._index.keys()):
                if k == namespace or k.startswith(f"{namespace}::"):
                    self._index[k].remove_ids(rem_set)
            return deleted

    def invalidate_expired(self) -> int:
        """Delete all expired entries across all namespaces.
        Returns the number of deleted rows.
        """
        with self._lock:
            self.flush()
            now = time.time()
            cur = self._raw_db.execute(
                "DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            self._raw_db.commit()
            deleted = cur.rowcount
            to_remove = [
                eid for eid, e in self._entries.items()
                if e.expires_at is not None and e.expires_at <= now
            ]
            for eid in to_remove:
                e = self._entries.pop(eid, None)
                if e:
                    self._exact_index.pop((e.namespace, e.query_hash, e.embedder), None)
            rem_set = set(to_remove)
            for vindex in self._index.values():
                vindex.remove_ids(rem_set)
            return deleted

    def clear(self, namespace: str | None = None) -> None:
        """Delete entries for *namespace* (or all namespaces if None)."""
        with self._lock:
            self.flush()
            if namespace is not None:
                _validate_namespace(namespace)
                self._raw_db.execute("DELETE FROM entries WHERE namespace=?", (namespace,))
                to_remove = [eid for eid, e in self._entries.items() if e.namespace == namespace]
                for eid in to_remove:
                    e = self._entries.pop(eid, None)
                    if e:
                        self._exact_index.pop((e.namespace, e.query_hash, e.embedder), None)
                self._index = {
                    k: v
                    for k, v in self._index.items()
                    if not (k == namespace or k.startswith(f"{namespace}::"))
                }
                self._entries_loaded = {
                    k
                    for k in self._entries_loaded
                    if not (k == namespace or k.startswith(f"{namespace}::"))
                }
            else:
                self._raw_db.execute("DELETE FROM entries")
                self._entries.clear()
                self._exact_index.clear()
                self._index.clear()
                self._entries_loaded.clear()
            self._raw_db.commit()

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------

    def size(self, namespace: str | None = None) -> int:
        """Return the number of (non-expired) entries."""
        now = time.time()
        self.flush()
        with self._lock:
            if self._raw_db is None:
                return 0
            if namespace is not None:
                _validate_namespace(namespace)
                return self._raw_db.execute(
                    """SELECT COUNT(*) FROM entries
                       WHERE namespace=?
                       AND (expires_at IS NULL OR expires_at > ?)""",
                    (namespace, now),
                ).fetchone()[0]
            return self._raw_db.execute(
                "SELECT COUNT(*) FROM entries WHERE expires_at IS NULL OR expires_at > ?",
                (now,),
            ).fetchone()[0]

    def namespace_stats(self, namespace: str) -> dict:
        """Return hit/miss statistics and entry count for a namespace."""
        _validate_namespace(namespace)
        now = time.time()
        self.flush()
        with self._lock:
            if self._raw_db is None:
                return {
                    "namespace": namespace,
                    "entries": 0,
                    "total_hits": 0,
                    "last_hit_at": None,
                }
            row = self._raw_db.execute(
                """SELECT COUNT(*), SUM(hits), MAX(last_hit_at)
                   FROM entries
                   WHERE namespace=? AND (expires_at IS NULL OR expires_at > ?)""",
                (namespace, now),
            ).fetchone()
            return {
                "namespace": namespace,
                "entries": row[0] or 0,
                "total_hits": row[1] or 0,
                "last_hit_at": row[2],
            }

    def close(self) -> None:
        """Close the database connection (safe to call more than once)."""
        self._close_db()

    def _close_db(self) -> None:
        stop_ev = getattr(self, "_stop_event", None)
        if stop_ev is not None:
            stop_ev.set()
        writer = getattr(self, "_writer_thread", None)
        if writer is not None and writer.is_alive():
            writer.join(timeout=0.2)

        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                self._flush_unlocked()
                raw_db = getattr(self, "_raw_db", None)
                if raw_db is not None:
                    raw_db.close()
                    self._raw_db = None
        else:
            raw_db = getattr(self, "_raw_db", None)
            if raw_db is not None:
                raw_db.close()
                self._raw_db = None

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._close_db()
        except Exception:
            pass
