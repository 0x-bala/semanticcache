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

    def execute(self, sql, *args):
        if sql.strip().lower() == "pragma journal_mode=wal":
            self.wal_attempts += 1
            if self.remaining_failures > 0:
                self.remaining_failures -= 1
                raise sqlite3.OperationalError("database is locked")
        return self._real.execute(sql, *args)

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

    class _BrokenWal(_LockedWalConnection):
        def execute(self, sql, *args):
            if sql.strip().lower() == "pragma journal_mode=wal":
                raise sqlite3.OperationalError("disk I/O error")
            return self._real.execute(sql, *args)

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _BrokenWal(real_connect(*a, **k), 0))
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        SQLiteStore(path=str(tmp_path / "broken.db"))


def test_wal_is_enabled_and_persists_for_second_opener(tmp_path):
    path = str(tmp_path / "wal.db")
    first = SQLiteStore(path=path)
    second = SQLiteStore(path=path)  # sees WAL already on and skips the switch
    assert second.db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    first.close()
    second.close()
