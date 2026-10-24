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

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .embedders import validate_embedding

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
                last_hit_at REAL
            )"""
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_ns ON entries(namespace)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS ix_ns_exp ON entries(namespace, expires_at)"
        )
        self.db.commit()

        # namespace -> (ids: list[int], matrix: np.ndarray[shape=(n, dim)])
        self._index: dict[str, tuple[list[int], np.ndarray]] = {}

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
                   (namespace, query, embedding, response, created_at, expires_at)
                   VALUES (?,?,?,?,?,?)""",
                (namespace, query, blob, response, now, now + ttl if ttl else None),
            )
            self.db.commit()
            entry_id: int = cur.lastrowid  # type: ignore[assignment]

            # Invalidate only the affected namespace, not the whole index
            self._index.pop(namespace, None)

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
    # Index loading (internal)
    # ------------------------------------------------------------------

    def _load(self, namespace: str) -> tuple[list[int], np.ndarray]:
        """Load (or return cached) the in-memory vector matrix for *namespace*.
        Caller must hold ``self._lock``.
        """
        if namespace in self._index:
            return self._index[namespace]

        rows = self.db.execute(
            """SELECT id, embedding FROM entries
               WHERE namespace=? AND (expires_at IS NULL OR expires_at > ?)""",
            (namespace, time.time()),
        ).fetchall()

        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for row_id, blob in rows:
            vec = _safe_load_blob(blob, self._embedding_dim)
            if vec is not None:
                ids.append(row_id)
                vecs.append(vec)

        if vecs:
            mat = np.vstack(vecs)  # shape (n, dim) — writeable copy
        else:
            dim = self._embedding_dim or 0
            mat = np.zeros((0, dim), dtype=np.float32)

        self._index[namespace] = (ids, mat)
        return ids, mat

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(self, namespace: str, embedding: np.ndarray) -> Entry | None:
        """Find the nearest stored entry by cosine similarity.

        Returns None when the namespace is empty, all entries have expired,
        or every embedding BLOB is corrupt.  Never recurses more than once
        (rebuilt index is always used for the second lookup).
        """
        _validate_namespace(namespace)
        embedding = validate_embedding(embedding, self._embedding_dim)

        with self._lock:
            ids, mat = self._load(namespace)
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
                self._index.pop(namespace, None)
                return None

            if row[2] is not None and row[2] <= time.time():
                # Best entry has expired; rebuild once and re-search
                self._index.pop(namespace, None)
                ids2, mat2 = self._load(namespace)
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
            self._index.pop(namespace, None)
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
                self._index.pop(namespace, None)  # only drop affected namespace
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
