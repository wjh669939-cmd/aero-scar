"""LLM capability seam: Service Definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    source: str = "provider"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
        }


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: LLMUsage
    raw: dict[str, Any] | None = None


class LLMProvider(Protocol):
    """One request in, one response out. Implementations are swappable plugins."""

    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        label: str = "",
        json_mode: bool = False,
    ) -> LLMResponse:
        ...
