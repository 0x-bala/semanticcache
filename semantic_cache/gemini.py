"""Optional Google Gemini adapters using the standard library HTTP client."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .embedders import validate_embedding


class GeminiAPIError(RuntimeError):
    """Raised when Gemini rejects a request or returns an unusable response."""


class GeminiClient:
    """Minimal Gemini text-generation client for use with ``SemanticCache``.

    The API key is read from ``GEMINI_API_KEY`` by default and is never logged
    or persisted.  Supply a key explicitly only from an application secret
    manager, never from source code.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY or pass api_key from a secret manager")
        self.timeout = timeout

    def _request(
        self, method: str, payload: dict[str, Any], max_retries: int = 3
    ) -> dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{method}?key={self.api_key}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                last_exc = exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(1.0 * (2**attempt))
                    continue
                raise GeminiAPIError(f"Gemini request failed: {exc}") from exc
            except (URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (2**attempt))
                    continue
                raise GeminiAPIError(f"Gemini request failed: {exc}") from exc
        raise GeminiAPIError(f"Gemini request failed after {max_retries} attempts: {last_exc}")

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """Generate a text response for a single prompt."""
        payload: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        response = self._request(f"{self.model}:generateContent", payload)
        try:
            parts = response["candidates"][0]["content"]["parts"]
            text = "".join(part["text"] for part in parts if "text" in part)
        except (IndexError, KeyError, TypeError) as exc:
            raise GeminiAPIError("Gemini returned no text candidate") from exc
        if not text:
            raise GeminiAPIError("Gemini returned an empty text candidate")
        return text


class GeminiEmbedder(GeminiClient):
    """Gemini semantic-similarity embedder compatible with ``SemanticCache``.

    Model specifications (``gemini-embedding-001``):
      - Default output dimension: **3072**
      - Supported range: **128 – 3072** (via Matryoshka Representation Learning)
      - Task type: ``SEMANTIC_SIMILARITY``
      - Lower dimensions (e.g. 768, 1536) trade slight quality for ~2–4×
        memory savings and faster cosine search.

    Parameters
    ----------
    model:
        Gemini embedding model name.  Default: ``gemini-embedding-001``.
    api_key:
        Gemini API key.  Read from ``GEMINI_API_KEY`` if not supplied.
    timeout:
        HTTP request timeout in seconds.
    output_dimensionality:
        Number of dimensions in the returned embedding vector.
        Must be between 128 and 3072 inclusive.  Default: **3072** (full
        resolution).  Use 768 or 1536 for reduced memory and faster search
        with minimal quality loss.
    """

    #: Minimum allowed output dimensionality for gemini-embedding-001.
    MIN_OUTPUT_DIM = 128
    #: Maximum (and default) output dimensionality for gemini-embedding-001.
    MAX_OUTPUT_DIM = 3072

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
        timeout: float = 30.0,
        output_dimensionality: int = 3072,
    ) -> None:
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        if not isinstance(output_dimensionality, int):
            raise TypeError(
                f"output_dimensionality must be an int, got {type(output_dimensionality).__name__}"
            )
        if not (self.MIN_OUTPUT_DIM <= output_dimensionality <= self.MAX_OUTPUT_DIM):
            raise ValueError(
                f"output_dimensionality must be between {self.MIN_OUTPUT_DIM} and "
                f"{self.MAX_OUTPUT_DIM}, got {output_dimensionality}"
            )
        self.output_dimensionality = output_dimensionality
        self.dim = output_dimensionality

    def embed(self, text: str) -> np.ndarray:
        payload: dict[str, Any] = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY",
            "outputDimensionality": self.output_dimensionality,
        }
        response = self._request(f"{self.model}:embedContent", payload)
        try:
            vector = np.asarray(response["embedding"]["values"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise GeminiAPIError("Gemini returned an invalid embedding") from exc
        vector = validate_embedding(vector)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector
