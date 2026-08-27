"""Typed-ish event bus. Mirrors DeepSeek Harness events without Cordis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

Listener = Callable[[dict[str, Any]], None]


class EventBus:
    """In-process pub/sub used by plugins and the research loop."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = defaultdict(list)

    def on(self, name: str, listener: Listener) -> Callable[[], None]:
        self._listeners[name].append(listener)

        def dispose() -> None:
            current = self._listeners.get(name, [])
            if listener in current:
                current.remove(listener)

        return dispose

    def emit(self, name: str, **payload: Any) -> dict[str, Any]:
        event = dict(payload)
        event.setdefault("event", name)
        for listener in list(self._listeners.get(name, [])):
            listener(event)
        return event
