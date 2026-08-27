"""OpenAI-compatible Chat Completions client (DeepSeek, OpenAI, local gateways)."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

from clh.config import LLMConfig
from clh.core.errors import LLMError
from clh.llm.provider import LLMResponse, LLMUsage


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text, including fenced markdown."""
    cleaned = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("model response did not contain a JSON object")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"failed to parse JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMError("JSON payload is not an object")
    return payload


class OpenAICompatLLM:
    name = "openai_compat"

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise LLMError("CLH_API_KEY / OPENAI_API_KEY is not configured")
        self.config = config
        kwargs: dict[str, Any] = {"api_key": config.api_key, "timeout": config.timeout_sec}
        if config.base_url:
            kwargs["base_url"] = config.base_url.rstrip("/")
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        system: str,
        user: str,
        *,
        label: str = "",
        json_mode: bool = False,
    ) -> LLMResponse:
        del label
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        attempts = max(1, self.config.retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_output_tokens,
                }
                if json_mode:
                    create_kwargs["response_format"] = {"type": "json_object"}
                completion = self._client.chat.completions.create(**create_kwargs)
                choice = completion.choices[0].message
                text = (choice.content or "").strip()
                usage = completion.usage
                record = LLMUsage(
                    model=self.config.model,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                )
                return LLMResponse(text=text, usage=record)
            except Exception as exc:
                last_error = exc
                if attempt >= attempts or not _is_transient(exc):
                    break
                delay = min(
                    self.config.retry_max_delay_sec,
                    self.config.retry_base_delay_sec * (2 ** (attempt - 1)),
                )
                time.sleep(delay)
        raise LLMError(f"LLM request failed: {last_error}") from last_error

    def ask_json(self, system: str, user: str, *, label: str = "") -> dict[str, Any]:
        instructed = user + "\n\nReturn a single JSON object. No markdown."
        try:
            response = self.complete(system, instructed, label=label, json_mode=True)
        except LLMError:
            response = self.complete(system, instructed, label=label, json_mode=False)
        return extract_json_object(response.text)


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = ("timeout", "rate", "429", "500", "502", "503", "529", "connection", "temporarily")
    return any(token in text for token in tokens)
