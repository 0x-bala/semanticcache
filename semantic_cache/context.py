"""Cache-context helpers for safe production cache partitioning."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheContext:
    """Fields that can change an LLM response and must partition its cache.

    ``namespace()`` returns an opaque, deterministic namespace.  It stores no
    raw tenant, user, prompt, or model values in the backing store, which keeps
    those identifiers out of database rows and ordinary cache logs.

    Include every field that affects an answer.  In particular, do not share a
    context across tenants, system-prompt revisions, tool configurations, or
    retrieval corpus revisions.
    """

    tenant: str
    model: str
    system_prompt_version: str
    user_scope: str = "shared"
    temperature: float | None = None
    tools_version: str = "none"
    retrieval_version: str = "none"
    locale: str = "und"
    prefix: str = "llm"

    def namespace(self) -> str:
        """Return an opaque namespace that is stable for this exact context."""
        values = {
            "tenant": self.tenant,
            "model": self.model,
            "system_prompt_version": self.system_prompt_version,
            "user_scope": self.user_scope,
            "temperature": self.temperature,
            "tools_version": self.tools_version,
            "retrieval_version": self.retrieval_version,
            "locale": self.locale,
        }
        string_values = (
            value for key, value in values.items() if key != "temperature"
        )
        if not all(isinstance(value, str) and value for value in string_values):
            raise ValueError("all string CacheContext fields must be non-empty strings")
        if self.temperature is not None and not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a number or None")
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValueError("prefix must be a non-empty string")

        encoded = json.dumps(
            values, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:32]
        return f"{self.prefix}:{digest}"
