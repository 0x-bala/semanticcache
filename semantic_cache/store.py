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

import hashlib
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .embedders import validate_embedding
from .guard import normalize_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
_MAX_NAMESPACE_LEN = 256
_MAX_QUERY_LEN = 32_768
_MAX_RESPONSE_LEN = 1_048_576   # 1 MB
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_namespace(ns: str) -> str:
    if not isinstance(ns, str) or not ns:
        raise ValueError("namespace must be a non-empty string")
    if len(ns) > _MAX_NAMESPACE_LEN:
        raise ValueError(
            f"namespace exceeds {_MAX_NAMESPACE_LEN} characters: {ns[:40]!r}..."
        )
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


# ---------------------------------------------------------------------------
# VectorIndex (pre-allocated doubling capacity index)
# ---------------------------------------------------------------------------

class VectorIndex:
    """In-memory vector index with capacity-doubling geometric growth.

    Pre-allocates an initial matrix capacity and doubles it whenever the
    capacity is reached, reducing per-insert allocations and copies from O(N)
    to O(1) amortized.
    """

    __slots__ = ("ids", "matrix", "count", "dim")

    def __init__(self, ids: list[int], matrix: np.ndarray, dim: int) -> None:
        self.ids = list(ids)
        self.count = len(self.ids)
        self.dim = dim
        capacity = max(16, self.count)
        if matrix.shape[0] < capacity:
            self.matrix = np.zeros((capacity, dim), dtype=np.float32)
            if self.count > 0:
                self.matrix[: self.count] = matrix[: self.count]
        else:
            self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)

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

    def active_matrix(self) -> np.ndarray:
        return self.matrix[: self.count]


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
    ) -> None:
        if embedding_dim is not None and embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer or None")
        if max_entries is not None and (not isinstance(max_entries, int) or max_entries < 1):
            raise ValueError("max_entries must be a positive integer or None")
        self._path = path
        self._embedding_dim = embedding_dim
        self._max_entries = max_entries
        self._lock = threading.Lock()

        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute(
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
        # Migrate existing tables if columns are missing
        existing_cols = {
            row[1] for row in self.db.execute("PRAGMA table_info(entries)").fetchall()
        }
        if "query_hash" not in existing_cols:
            self.db.execute("ALTER TABLE entries ADD COLUMN query_hash TEXT")
        if "embedder" not in existing_cols:
            self.db.execute("ALTER TABLE entries ADD COLUMN embedder TEXT NOT NULL DEFAULT ''")

        # Backfill query_hash for legacy rows where missing
        unhashed = self.db.execute(
            "SELECT id, query FROM entries WHERE query_hash IS NULL OR query_hash = ''"
        ).fetchall()
        if unhashed:
            updates = [
                (hashlib.sha256(normalize_text(q).encode("utf-8")).hexdigest(), rid)
                for rid, q in unhashed
            ]
            self.db.executemany("UPDATE entries SET query_hash = ? WHERE id = ?", updates)

        self.db.execute("CREATE INDEX IF NOT EXISTS ix_ns ON entries(namespace)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS ix_ns_exp ON entries(namespace, expires_at)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS ix_ns_hash ON entries(namespace, query_hash)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS ix_ns_emb ON entries(namespace, embedder)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_exact "
            "ON entries(namespace, query_hash, embedder)"
        )
        self.db.commit()

        # index_key -> VectorIndex
        self._index: dict[str, VectorIndex] = {}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

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
        """Insert a new cache entry.  Returns the new entry ID."""
        _validate_namespace(namespace)
        _validate_ttl(ttl)
        embedding = validate_embedding(embedding, self._embedding_dim)

        if len(query.encode("utf-8")) > _MAX_QUERY_LEN:
            raise ValueError("query exceeds maximum allowed size")
        if len(response.encode("utf-8")) > _MAX_RESPONSE_LEN:
            raise ValueError("response exceeds maximum allowed size")

        now = time.time()
        blob = embedding.astype(np.float32).tobytes()

        with self._lock:
            cur = self.db.execute(
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
                    now + ttl if ttl else None,
                    query_hash,
                    embedder,
                ),
            )
            self.db.commit()
            entry_id: int = cur.lastrowid  # type: ignore[assignment]

            # Incremental index update: append vector using geometric doubling growth
            index_keys = [namespace]
            if embedder:
                index_keys.append(f"{namespace}::{embedder}")

            for ikey in index_keys:
                if ikey in self._index:
                    self._index[ikey].append(entry_id, embedding)

            if self._max_entries is not None:
                self._evict_lru_if_needed()

        logger.debug("cache.add namespace=%r id=%d", namespace, entry_id)
        return entry_id

    def _evict_lru_if_needed(self) -> None:
        """Evict the least-recently-used entry if over the entry limit.
        Caller must hold ``self._lock``.
        """
        total = self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if total > self._max_entries:
            # LRU: order by last_hit_at NULLS FIRST (never hit), then created_at
            self.db.execute(
                """DELETE FROM entries WHERE id IN (
                    SELECT id FROM entries
                    ORDER BY last_hit_at ASC NULLS FIRST, created_at ASC
                    LIMIT ?
                )""",
                (total - self._max_entries,),
            )
            self.db.commit()
            self._index.clear()

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
            if embedder:
                cur = self.db.execute(
                    """SELECT id, query, response, expires_at FROM entries
                       WHERE namespace=? AND query_hash=? AND embedder=?
                         AND (expires_at IS NULL OR expires_at > ?)
                       ORDER BY id DESC LIMIT 1""",
                    (namespace, query_hash, embedder, now),
                )
            else:
                cur = self.db.execute(
                    """SELECT id, query, response, expires_at FROM entries
                       WHERE namespace=? AND query_hash=?
                         AND (expires_at IS NULL OR expires_at > ?)
                       ORDER BY id DESC LIMIT 1""",
                    (namespace, query_hash, now),
                )
            row = cur.fetchone()
            if row is None:
                return None
            return Entry(id=row[0], query=row[1], response=row[2], score=1.0)

    # ------------------------------------------------------------------
    # Index loading (internal)
    # ------------------------------------------------------------------

    def _load(self, namespace: str, embedder: str | None = None) -> VectorIndex:
        """Load (or return cached) the in-memory vector index for *namespace*.
        Caller must hold ``self._lock``.
        """
        index_key = f"{namespace}::{embedder}" if embedder else namespace
        if index_key in self._index:
            return self._index[index_key]

        now = time.time()
        if embedder:
            rows = self.db.execute(
                """SELECT id, embedding FROM entries
                   WHERE namespace=? AND embedder=?
                     AND (expires_at IS NULL OR expires_at > ?)""",
                (namespace, embedder, now),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT id, embedding FROM entries
                   WHERE namespace=? AND (expires_at IS NULL OR expires_at > ?)""",
                (namespace, now),
            ).fetchall()

        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for row_id, blob in rows:
            vec = _safe_load_blob(blob, self._embedding_dim)
            if vec is not None:
                ids.append(row_id)
                vecs.append(vec)

        if vecs:
            initial_mat = np.vstack(vecs)
            dim = initial_mat.shape[1]
        else:
            dim = self._embedding_dim or 0
            initial_mat = np.zeros((0, dim), dtype=np.float32)

        vindex = VectorIndex(ids, initial_mat, dim)
        self._index[index_key] = vindex
        return vindex

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(
        self,
        namespace: str,
        embedding: np.ndarray,
        embedder: str | None = None,
    ) -> Entry | None:
        """Find the nearest stored entry by cosine similarity.

        Returns None when the namespace is empty, all entries have expired,
        or every embedding BLOB is corrupt.  Never recurses more than once
        (rebuilt index is always used for the second lookup).
        """
        _validate_namespace(namespace)
        embedding = validate_embedding(embedding, self._embedding_dim)
        index_key = f"{namespace}::{embedder}" if embedder else namespace

        with self._lock:
            vindex = self._load(namespace, embedder)
            ids, mat = vindex.ids, vindex.active_matrix()
            if not ids:
                return None

            scores = mat @ embedding  # cosine similarity (unit vecs)
            best_idx = int(np.argmax(scores))
            best_id = ids[best_idx]
            best_score = float(scores[best_idx])

            row = self.db.execute(
                "SELECT query, response, expires_at FROM entries WHERE id=?",
                (best_id,),
            ).fetchone()

            if row is None:
                # Row was deleted between index build and now
                self._index.pop(index_key, None)
                return None

            if row[2] is not None and row[2] <= time.time():
                # Best entry has expired; rebuild once and re-search
                self._index.pop(index_key, None)
                vindex2 = self._load(namespace, embedder)
                ids2, mat2 = vindex2.ids, vindex2.active_matrix()
                if not ids2:
                    return None
                scores2 = mat2 @ embedding
                best_idx2 = int(np.argmax(scores2))
                best_id = ids2[best_idx2]
                best_score = float(scores2[best_idx2])
                row = self.db.execute(
                    "SELECT query, response, expires_at FROM entries WHERE id=?",
                    (best_id,),
                ).fetchone()
                if row is None or (row[2] is not None and row[2] <= time.time()):
                    return None

            return Entry(id=best_id, query=row[0], response=row[1], score=best_score)

    def record_hit(self, entry_id: int) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE entries SET hits = hits + 1, last_hit_at = ? WHERE id=?",
                (time.time(), entry_id),
            )
            self.db.commit()

    # ------------------------------------------------------------------
    # Deletion / maintenance
    # ------------------------------------------------------------------

    def delete_by_query(self, namespace: str, query: str) -> int:
        """Delete all entries whose stored query exactly matches *query*.
        Returns the number of deleted rows.
        """
        _validate_namespace(namespace)
        with self._lock:
            cur = self.db.execute(
                "DELETE FROM entries WHERE namespace=? AND query=?",
                (namespace, query),
            )
            self.db.commit()
            self._index = {
                k: v for k, v in self._index.items()
                if not (k == namespace or k.startswith(f"{namespace}::"))
            }
            return cur.rowcount

    def invalidate_expired(self) -> int:
        """Delete all expired entries across all namespaces.
        Returns the number of deleted rows.
        """
        with self._lock:
            cur = self.db.execute(
                "DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (time.time(),),
            )
            self.db.commit()
            self._index.clear()
            return cur.rowcount

    def clear(self, namespace: str | None = None) -> None:
        """Delete entries for *namespace* (or all namespaces if None)."""
        with self._lock:
            if namespace is not None:
                _validate_namespace(namespace)
                self.db.execute(
                    "DELETE FROM entries WHERE namespace=?", (namespace,)
                )
                self._index = {
                    k: v for k, v in self._index.items()
                    if not (k == namespace or k.startswith(f"{namespace}::"))
                }
            else:
                self.db.execute("DELETE FROM entries")
                self._index.clear()
            self.db.commit()

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------

    def size(self, namespace: str | None = None) -> int:
        """Return the number of (non-expired) entries."""
        now = time.time()
        if namespace is not None:
            _validate_namespace(namespace)
            return self.db.execute(
                """SELECT COUNT(*) FROM entries
                   WHERE namespace=?
                   AND (expires_at IS NULL OR expires_at > ?)""",
                (namespace, now),
            ).fetchone()[0]
        return self.db.execute(
            "SELECT COUNT(*) FROM entries WHERE expires_at IS NULL OR expires_at > ?",
            (now,),
        ).fetchone()[0]

    def namespace_stats(self, namespace: str) -> dict:
        """Return hit/miss statistics and entry count for a namespace."""
        _validate_namespace(namespace)
        now = time.time()
        row = self.db.execute(
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
        """Close the database connection."""
        with self._lock:
            self.db.close()
