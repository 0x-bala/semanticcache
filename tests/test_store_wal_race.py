"""Regression tests for the "database is locked" race when opening a shared cache file.

Two processes opening the same SQLite file at once could make
``PRAGMA journal_mode=WAL`` fail immediately with "database is locked", which
crashed ``SQLiteStore.__init__`` (seen as a random CI failure).
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

from semantic_cache.store import SQLiteStore


class _LockedWalConnection:
    """Wraps a real connection; the first *fail_times* WAL switches report a lock."""

    def __init__(self, real: sqlite3.Connection, fail_times: int) -> None:
        self._real = real
        self.remaining_failures = fail_times
        self.wal_attempts = 0
        self.closed = False

    def execute(self, sql, *args):
        if sql.strip().lower() == "pragma journal_mode=wal":
            self.wal_attempts += 1
            if self.remaining_failures > 0:
                self.remaining_failures -= 1
                raise sqlite3.OperationalError("database is locked")
        return self._real.execute(sql, *args)

    def close(self):
        self.closed = True
        return self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _patch_connect(monkeypatch, fail_times: int) -> list[_LockedWalConnection]:
    real_connect = sqlite3.connect
    made: list[_LockedWalConnection] = []

    def fake_connect(*args, **kwargs):
        conn = _LockedWalConnection(real_connect(*args, **kwargs), fail_times)
        made.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    monkeypatch.setattr("semantic_cache.store.time.sleep", lambda _s: None)
    return made


def test_store_retries_when_wal_switch_is_locked(tmp_path, monkeypatch):
    made = _patch_connect(monkeypatch, fail_times=3)
    store = SQLiteStore(path=str(tmp_path / "shared.db"))
    assert made[0].wal_attempts == 4  # three locked attempts, then success
    assert store.size() == 0
    store.close()


def test_store_still_opens_if_wal_never_succeeds(tmp_path, monkeypatch, caplog):
    _patch_connect(monkeypatch, fail_times=10_000)
    with caplog.at_level(logging.WARNING):
        store = SQLiteStore(path=str(tmp_path / "busy.db"))
    assert store.size() == 0  # fell back to the default journal mode
    assert any("WAL" in record.getMessage() for record in caplog.records)
    store.close()


def test_other_operational_errors_are_not_swallowed(tmp_path, monkeypatch):
    real_connect = sqlite3.connect
    made: list[_LockedWalConnection] = []

    class _BrokenWal(_LockedWalConnection):
        def execute(self, sql, *args):
            if sql.strip().lower() == "pragma journal_mode=wal":
                raise sqlite3.OperationalError("disk I/O error")
            return self._real.execute(sql, *args)

    def fake_connect(*a, **k):
        conn = _BrokenWal(real_connect(*a, **k), 0)
        made.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        SQLiteStore(path=str(tmp_path / "broken.db"))

    # Verify that the connection was closed during initialization failure cleanup
    assert len(made) == 1
    assert made[0].closed is True


def test_init_failure_does_not_mask_original_exception_when_close_fails(tmp_path, monkeypatch):
    """If close() raises an exception during cleanup, the original exception is preserved."""
    real_connect = sqlite3.connect

    class _FailingCloseConn:
        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def execute(self, sql, *args):
            if "CREATE TABLE" in sql.upper():
                raise RuntimeError("original table error")
            return self._real.execute(sql, *args)

        def close(self):
            self._real.close()
            raise RuntimeError("close error")

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: _FailingCloseConn(real_connect(*a, **k))
    )

    with pytest.raises(RuntimeError, match="original table error"):
        SQLiteStore(path=str(tmp_path / "failing_cleanup.db"))


def test_migration_retry_exhaustion_fails_explicitly(tmp_path, monkeypatch):
    """When BEGIN IMMEDIATE is permanently locked/busy, migration exhausts retries and fails."""
    monkeypatch.setattr("semantic_cache.store.time.sleep", lambda _s: None)

    real_connect = sqlite3.connect
    conns = []

    class _LockedMigrationConn:
        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real
            self.closed = False

        def execute(self, sql, *args):
            if sql.strip().upper() == "BEGIN IMMEDIATE":
                raise sqlite3.OperationalError("database is locked")
            return self._real.execute(sql, *args)

        def close(self):
            self.closed = True
            return self._real.close()

        def __getattr__(self, name):
            return getattr(self._real, name)

    def fake_connect(*args, **kwargs):
        conn = _LockedMigrationConn(real_connect(*args, **kwargs))
        conns.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", fake_connect)

    with pytest.raises(
        sqlite3.OperationalError, match="Could not acquire migration lock after 40 attempts"
    ):
        SQLiteStore(path=str(tmp_path / "locked_mig.db"))

    assert len(conns) == 1
    assert conns[0].closed is True


def test_store_context_manager_and_explicit_cleanup(tmp_path):
    path = str(tmp_path / "ctx.db")
    with SQLiteStore(path=path) as store:
        assert store.size() == 0
        raw_db = store.db

    # Exiting context manager closes the connection
    with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
        raw_db.execute("SELECT 1")

    # Double close is safe and idempotent
    store.close()


def test_wal_is_enabled_and_persists_for_second_opener(tmp_path):
    path = str(tmp_path / "wal.db")
    first = SQLiteStore(path=path)
    second = SQLiteStore(path=path)  # sees WAL already on and skips the switch
    assert second.db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    first.close()
    second.close()


def test_concurrent_wal_operations(tmp_path):
    """Concurrent readers and writers under WAL mode succeed without deadlocks."""
    import concurrent.futures

    import numpy as np

    path = str(tmp_path / "concurrent_wal.db")
    # Initialize the database file
    init_store = SQLiteStore(path=path, embedding_dim=4)
    init_store.close()

    errors = []

    def worker(worker_id: int):
        try:
            with SQLiteStore(path=path, embedding_dim=4) as store:
                vec = np.zeros(4, dtype=np.float32)
                vec[worker_id % 4] = 1.0
                for i in range(5):
                    store.add(
                        f"ns_{worker_id}", f"q_{worker_id}_{i}", vec, f"r_{worker_id}_{i}", None
                    )
                    assert store.size(f"ns_{worker_id}") >= 1
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, i) for i in range(4)]
        concurrent.futures.wait(futures)

    assert not errors
    with SQLiteStore(path=path, embedding_dim=4) as final_store:
        total = sum(final_store.size(f"ns_{i}") for i in range(4))
        assert total == 20

