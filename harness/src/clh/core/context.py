"""Shared runtime context. Simplified Cordis: plugins provide named services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

from clh.config import HarnessConfig
from clh.core.events import EventBus, Listener

T = TypeVar("T")


class HarnessContext:
    """Composition root for one research run.

    Services are keyed by name (``llm``, ``tools``, ``session``, ``evaluator``).
    A plugin may replace a service; the research loop never imports providers
    directly.
    """

    def __init__(
        self,
        config: HarnessConfig,
        run_dir: Path,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.workspace_root = workspace_root or Path.cwd()
        self.events = EventBus()
        self._services: dict[str, Any] = {}

    def provide(self, key: str, service: Any) -> None:
        self._services[key] = service

    def get(self, key: str, expected: type[T] | None = None) -> T:
        if key not in self._services:
            raise KeyError(f"service {key!r} is not registered")
        service = self._services[key]
        if expected is not None and not isinstance(service, expected):
            raise TypeError(f"service {key!r} is {type(service).__name__}, expected {expected.__name__}")
        return service

    def maybe(self, key: str) -> Any | None:
        return self._services.get(key)

    def on(self, name: str, listener: Listener) -> Callable[[], None]:
        return self.events.on(name, listener)

    def emit(self, name: str, **payload: Any) -> dict[str, Any]:
        return self.events.emit(name, **payload)
