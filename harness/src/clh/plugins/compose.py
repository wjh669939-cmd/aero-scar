"""Compose a run from plugins. Profiles stack like dsh bundles."""

from __future__ import annotations

from pathlib import Path

from clh.config import HarnessConfig
from clh.core.context import HarnessContext
from clh.core.session import SessionLog
from clh.core.tools import Tool, ToolRegistry
from clh.research.loop import build_llm


def boot_context(config: HarnessConfig, run_dir: Path, *, workspace_root: Path | None = None) -> HarnessContext:
    ctx = HarnessContext(config, run_dir, workspace_root=workspace_root)
    ctx.provide("session", SessionLog(run_dir / "session.jsonl"))
    ctx.provide("llm", build_llm(config))
    tools = ToolRegistry()
    session: SessionLog = ctx.get("session", SessionLog)
    tools.register(
        Tool(
            name="read_lineage",
            description="Read model-visible research lineage. Hidden test scores are never included.",
            permission="read_only",
            handler=lambda _args: {"events": session.model_visible()[-20:]},
        )
    )
    allowed_sources = [
        "pretrain_train",
        "matched_climate",
        "shifted_climate",
        "leak_val",
        "matched_ZBHH",
        "shifted_ZJHK",
        "same_source_leak",
    ]

    def _write_external_data(args: dict) -> dict:
        source_id = str(args.get("source_id") or "")
        if source_id not in allowed_sources:
            return {
                "ok": False,
                "reason": "unknown or forbidden source; evaluator catalog only",
                "allowed": allowed_sources,
            }
        session.append("write_external_data", source_id=source_id)
        return {
            "ok": True,
            "source_id": source_id,
            "note": "declare this id in external_manifest.json; the evaluator owns filtering and merge",
        }

    tools.register(
        Tool(
            name="write_external_data",
            description=(
                "Declare one catalogued extra-evidence source for the data axis. "
                "Cannot read sealed/, ZBAD, or pretrain/test. Merge is evaluator-owned."
            ),
            permission="write",
            schema={
                "type": "object",
                "properties": {"source_id": {"type": "string"}},
                "required": ["source_id"],
            },
            handler=_write_external_data,
        )
    )
    ctx.provide("tools", tools)
    return ctx
