"""Axis specialists: hypothesis then executable action."""

from __future__ import annotations

from pathlib import Path

from clh.config import AxisName
from clh.core.session import format_lineage_for_prompt
from clh.llm.openai_compat import extract_json_object
from clh.llm.provider import LLMProvider
from clh.research.axis_lock import allowed_files
from clh.research.cards import ActionCard, HypothesisCard
from clh.research.presets import preset_files

HYPOTHESIS_SYSTEM = """You are a research specialist for closed-loop Auto Research in aerodrome weather / ATC.
Propose ONE falsifiable hypothesis on the assigned axis. Do not write code yet.
Return JSON with keys: axis, claim, mechanism, target_slice, expected_gain, side_effects, falsification, negative_control.
The hypothesis must name a failure slice, a mechanism, and a falsification condition.
Never request test labels or hidden certification splits.
"""

ACTION_SYSTEM = """You convert an approved hypothesis into a single-axis intervention.
You may only change files on the assigned axis.
Prefer a named preset when it matches the claim.
The data axis may only name evaluator-catalog sources (pretrain_train, matched_climate, shifted_climate, leak_val). Never request sealed/, ZBAD, or pretrain/test.
Return JSON: {"axis": "...", "kind": "preset"|"files", "preset": "runway_wind|ridge|extreme_wind_weights|matched_airport|shifted_airport|same_source_leak", "files": {}, "notes": ""}.
If kind is files, keys must be allowed filenames and values the full file contents.
"""


class Specialist:
    def __init__(self, llm: LLMProvider, *, domain: str = "dummy") -> None:
        self.llm = llm
        self.domain = domain

    def hypothesize(
        self,
        axis: AxisName,
        lineage_rows: list[dict],
        *,
        parent_trial_id: str | None = None,
    ) -> HypothesisCard:
        user = (
            f"Assigned axis: {axis}\n"
            f"Allowed files: {sorted(allowed_files(axis))}\n"
            f"Recent lineage:\n{format_lineage_for_prompt(lineage_rows)}\n"
        )
        payload = _ask_json(self.llm, HYPOTHESIS_SYSTEM, user, label="hypothesis")
        payload["axis"] = axis
        payload["parent_trial_id"] = parent_trial_id
        return HypothesisCard.model_validate(payload)

    def act(self, hypothesis: HypothesisCard, pipeline_root: Path) -> ActionCard:
        user = (
            f"{hypothesis.to_prompt()}\n"
            f"Allowed files: {sorted(allowed_files(hypothesis.axis))}\n"
            f"Current files:\n{_snapshot(pipeline_root, hypothesis.axis)}\n"
        )
        payload = _ask_json(self.llm, ACTION_SYSTEM, user, label="action")
        payload["axis"] = hypothesis.axis
        payload["preset"] = payload.get("preset") or ""
        payload["notes"] = payload.get("notes") or ""
        payload["kind"] = payload.get("kind") or "preset"
        payload["files"] = payload.get("files") or {}
        action = ActionCard.model_validate(payload)
        preset = action.preset or _default_preset(hypothesis.axis)
        if action.kind == "preset" or not action.files:
            action = ActionCard(
                axis=hypothesis.axis,
                kind="preset",
                preset=preset,
                files=preset_files(hypothesis.axis, preset, pipeline_root, domain=self.domain),
                notes=action.notes or preset,
            )
        return action


def _default_preset(axis: AxisName) -> str:
    return {
        "representation": "runway_wind",
        "model": "ridge",
        "physics": "extreme_wind_weights",
        "objective": "extreme_wind_weights",
        "data": "matched_airport",
    }[axis]


def _ask_json(llm: LLMProvider, system: str, user: str, *, label: str) -> dict:
    ask_json = getattr(llm, "ask_json", None)
    if callable(ask_json):
        return ask_json(system, user, label=label)
    response = llm.complete(system, user + "\nReturn JSON only.", label=label, json_mode=True)
    return extract_json_object(response.text)


def _snapshot(pipeline_root: Path, axis: AxisName) -> str:
    chunks: list[str] = []
    for name in sorted(allowed_files(axis)):
        path = pipeline_root / name
        if path.exists():
            chunks.append(f"--- {name} ---\n{path.read_text(encoding='utf-8')}")
    return "\n".join(chunks) or "(empty)"
