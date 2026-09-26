"""SQLite-backed storage with an in-memory vector index for fast search."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Entry:
    id: int
    query: str
    response: str
    score: float = 0.0


class SQLiteStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                query TEXT NOT NULL,
                embedding BLOB NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                hits INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_ns ON entries(namespace)")
        self.db.commit()
        self._index: dict[str, tuple[list[int], np.ndarray]] = {}  # namespace -> (ids, matrix)

    def add(self, namespace: str, query: str, embedding: np.ndarray, response: str, ttl: float | None) -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO entries (namespace, query, embedding, response, created_at, expires_at) VALUES (?,?,?,?,?,?)",
            (namespace, query, embedding.astype(np.float32).tobytes(), response, now, now + ttl if ttl else None),
        )
        self.db.commit()
        self._index.pop(namespace, None)  # invalidate
        return cur.lastrowid

    def _load(self, namespace: str) -> tuple[list[int], np.ndarray]:
        if namespace not in self._index:
            rows = self.db.execute(
                "SELECT id, embedding FROM entries WHERE namespace=? AND (expires_at IS NULL OR expires_at>?)",
                (namespace, time.time()),
            ).fetchall()
            ids = [r[0] for r in rows]
            mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 0), np.float32)
            self._index[namespace] = (ids, mat)
        return self._index[namespace]

    def search(self, namespace: str, embedding: np.ndarray) -> Entry | None:
        ids, mat = self._load(namespace)
        if not ids:
            return None
        scores = mat @ embedding  # cosine similarity (vectors are unit length)
        best = int(np.argmax(scores))
        row = self.db.execute(
            "SELECT query, response, expires_at FROM entries WHERE id=?", (ids[best],)
        ).fetchone()
        if row[2] is not None and row[2] <= time.time():  # expired since index was built
            self._index.pop(namespace, None)
            return self.search(namespace, embedding)
        return Entry(id=ids[best], query=row[0], response=row[1], score=float(scores[best]))

    def record_hit(self, entry_id: int) -> None:
        self.db.execute("UPDATE entries SET hits = hits + 1 WHERE id=?", (entry_id,))
        self.db.commit()

    def clear(self, namespace: str | None = None) -> None:
        if namespace:
            self.db.execute("DELETE FROM entries WHERE namespace=?", (namespace,))
        else:
            self.db.execute("DELETE FROM entries")
        self.db.commit()
        self._index.clear()

    def size(self, namespace: str | None = None) -> int:
        if namespace:
            return self.db.execute("SELECT COUNT(*) FROM entries WHERE namespace=?", (namespace,)).fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
