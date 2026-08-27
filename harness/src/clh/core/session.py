"""Append-only session log. Model-visible facts must be reconstructable from it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionLog:
    """Durable JSONL log for one run. Source of lineage and model context."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, event_type: str, **payload: Any) -> dict[str, Any]:
        record = {
            "ts": utcnow_iso(),
            "type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [row for row in self.load() if row.get("type") == event_type]

    def model_visible(self) -> list[dict[str, Any]]:
        """Facts the research agent is allowed to see. Test labels never appear."""
        hidden = {"certification_score", "test_labels", "certification"}
        return [row for row in self.load() if row.get("type") not in hidden]


def format_lineage_for_prompt(rows: Iterable[dict[str, Any]], *, limit: int = 12) -> str:
    material = [row for row in rows if row.get("type") in {"trial_result", "hypothesis", "evidence_gate"}]
    tail = material[-limit:]
    if not tail:
        return "(empty lineage)"
    return json.dumps(tail, ensure_ascii=False, indent=2)
