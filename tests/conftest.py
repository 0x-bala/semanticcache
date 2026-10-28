"""Shared pytest fixtures.

The autouse fixture below pins ``OLLAMA_HOST`` to a closed port (127.0.0.1:9)
for every test, so no test can ever reach a real Ollama server by accident.
Tests that need an Ollama client must mock it explicitly.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _no_real_ollama(monkeypatch):
    """Point every test at a closed port unless the test sets OLLAMA_HOST itself."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    # Make sure a stale process-level value cannot override the monkeypatched one.
    os.environ.pop("OLLAMA_HOST", None)
    yield