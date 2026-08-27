"""Plugin protocol. Everything in the harness is a plugin, as in dsh."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from clh.core.context import HarnessContext


@runtime_checkable
class Plugin(Protocol):
    """A capability that registers services, tools, or listeners on the context."""

    name: str

    def apply(self, ctx: HarnessContext) -> None:
        """Register this plugin's services onto ``ctx``."""
