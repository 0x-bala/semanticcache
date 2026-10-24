"""LLM-based semantic equivalence verifier for SemanticCache.

Uses a Gemini generative model (temperature 0) to decide whether two
questions would produce the same correct answer.  Results are cached
per ``(query_a, query_b, model)`` triple to avoid repeat API calls.

Security notes
--------------
- Queries are sent to the Gemini API for verification.  Do not use with
  queries that contain sensitive PII unless your Gemini data-handling
  agreement permits it.
- The verifier never logs query text unless ``log_queries=True``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class GeminiVerifier:
    """Verify whether two queries are semantically equivalent using Gemini.

    Parameters
    ----------
    model:
        Gemini model for verification.  Default: ``gemini-2.0-flash``.
    api_key:
        Gemini API key.  Read from ``GEMINI_API_KEY`` if not supplied.
    timeout:
        HTTP request timeout in seconds per attempt.
    max_retries:
        Maximum retries on transient errors (429, 5xx, network).
    safe_default:
        Value returned on any error (API failure, timeout, parse error).
        Default: ``False`` (reject on error — fail-safe).
    log_queries:
        When ``True``, query text is logged at DEBUG level.
        **Never enable in production.**
    """

    SYSTEM_PROMPT = (
        "You are a strict semantic equivalence judge. "
        "Answer ONLY 'yes' or 'no'. "
        "Do not explain, qualify, or add any other text."
    )

    USER_PROMPT_TEMPLATE = (
        "Would these two questions have the same correct answer?\n\n"
        "Question A: {query_a}\n"
        "Question B: {query_b}\n\n"
        "Answer (yes or no):"
    )

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        timeout: float = 15.0,
        max_retries: int = 2,
        safe_default: bool = False,
        log_queries: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY or pass api_key from a secret manager")
        self.timeout = timeout
        self.max_retries = max_retries
        self.safe_default = safe_default
        self._log_queries = log_queries
        self.last_error: Exception | None = None
        # Thread-safe verdict cache: (cache_key) -> (bool, timestamp)
        self._cache: dict[str, tuple[bool, float]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(query_a: str, query_b: str, model: str) -> str:
        """Deterministic cache key for a (query_a, query_b, model) triple.

        The key is order-independent: verify(A, B) == verify(B, A).
        """
        pair = tuple(sorted([query_a.strip().lower(), query_b.strip().lower()]))
        raw = f"{pair[0]}||{pair[1]}||{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _call_api(self, query_a: str, query_b: str) -> bool:
        """Make the Gemini API call with retries.  Returns bool or raises."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        prompt = self.USER_PROMPT_TEMPLATE.format(query_a=query_a, query_b=query_b)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": self.SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 8,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urlopen(request, timeout=self.timeout) as resp:
                    body: dict[str, Any] = json.load(resp)
                candidates = body.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates returned from Gemini API")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = ""
                for p in parts:
                    if isinstance(p, dict) and "text" in p:
                        text = p["text"].strip().lower()
                        break
                if not text:
                    raise ValueError("No text part found in candidate response")
                self.last_error = None
                return text.startswith("yes")
            except HTTPError as exc:
                last_exc = exc
                self.last_error = exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    sleep_s = 2.0 * (2**attempt)
                    if exc.code == 429:
                        try:
                            raw_body = exc.read().decode("utf-8")
                            err_body = json.loads(raw_body)
                            for detail in err_body.get("error", {}).get("details", []):
                                if "retryDelay" in detail:
                                    raw_delay = detail["retryDelay"].rstrip("s")
                                    sleep_s = max(sleep_s, float(raw_delay) + 1.0)
                        except Exception:
                            sleep_s = max(sleep_s, 5.0)
                    time.sleep(sleep_s)
                    continue
                logger.warning("GeminiVerifier HTTP error: %s", exc)
                return self.safe_default
            except (URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                self.last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (2**attempt))
                    continue
                logger.warning("GeminiVerifier network error: %s", exc)
                return self.safe_default
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                last_exc = exc
                self.last_error = exc
                logger.warning("GeminiVerifier parse error: %s", exc)
                return self.safe_default

        self.last_error = last_exc
        logger.warning("GeminiVerifier failed after %d attempts: %s", self.max_retries, last_exc)
        return self.safe_default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, query_a: str, query_b: str) -> bool:
        """Return ``True`` if the two queries are semantically equivalent.

        Results are cached per ``(query_a, query_b, model)`` triple so
        repeated checks for the same pair are free.
        """
        key = self._cache_key(query_a, query_b, self.model)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key][0]

        if self._log_queries:
            logger.debug("GeminiVerifier.verify query_a=%r query_b=%r", query_a, query_b)

        result = self._call_api(query_a, query_b)

        with self._cache_lock:
            self._cache[key] = (result, time.time())

        return result

    def __call__(self, query_a: str, query_b: str) -> bool:
        """Callable interface for use as ``SemanticCache(verifier=...)``."""
        return self.verify(query_a, query_b)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    @property
    def cache_size(self) -> int:
        """Number of cached verdicts."""
        with self._cache_lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the verdict cache."""
        with self._cache_lock:
            self._cache.clear()
