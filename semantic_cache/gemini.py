"""Optional Google Gemini adapters using the standard library HTTP client."""
from __future__ import annotations

import json
import os
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

    def _request(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{method}?key={self.api_key}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise GeminiAPIError(f"Gemini request failed: {exc}") from exc

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
    """Gemini semantic-similarity embedder compatible with ``SemanticCache``."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(model=model, api_key=api_key, timeout=timeout)

    def embed(self, text: str) -> np.ndarray:
        response = self._request(
            f"{self.model}:embedContent",
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "taskType": "SEMANTIC_SIMILARITY",
            },
        )
        try:
            vector = np.asarray(response["embedding"]["values"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise GeminiAPIError("Gemini returned an invalid embedding") from exc
        vector = validate_embedding(vector)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector
