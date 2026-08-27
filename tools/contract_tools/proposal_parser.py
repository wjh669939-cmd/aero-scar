"""A-4（部分）：把 LLM 臂的提案文本解析为符合 trial schema 的记录。

原则（自 simple_ar/aerowf_v1 移植，正典位置在此）：
- 严格解析：只接受单个 JSON 对象，不做自由文本打捞；
- 科学字段（hypothesis / evidence_anchor / falsification）无静默默认值；
- 引用模板动作时不继承注册表文本——模型必须复述自己的假设，
  偷懒复读在 lineage 里可见；
- 解析成功的记录仍要过冻结 schema 校验，双保险。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from contract_tools.validate import ContractViolation, validate_trial

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_REQUIRED_FROM_LLM = (
    "axis",
    "action_id",
    "hypothesis",
    "evidence_anchor",
    "target_slices",
    "falsification",
    "editable_paths",
)

_AXIS_SHORT = {
    "representation": "rep",
    "objective_tier1": "obj",
    "objective_tier2": "obj",
    "model": "model",
}


@dataclass(frozen=True)
class ParsedProposal:
    ok: bool
    trial_record: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def parse_llm_proposal(
    raw_text: str,
    trial_seq: int,
    parent_trial: str,
    gpu_hours_cap: float = 1.0,
    screening_seed: int = 42,
) -> ParsedProposal:
    match = _JSON_BLOCK.search(raw_text)
    if not match:
        return ParsedProposal(ok=False, errors=["no JSON object found in LLM output"])
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return ParsedProposal(ok=False, errors=[f"JSON parse error: {exc}"])

    missing = [k for k in _REQUIRED_FROM_LLM if not payload.get(k)]
    if missing:
        return ParsedProposal(ok=False, errors=[f"missing required fields: {missing}"])

    axis = payload["axis"]
    prefix = _AXIS_SHORT.get(axis)
    if prefix is None:
        return ParsedProposal(ok=False, errors=[f"unknown axis: {axis}"])

    action_id = str(payload["action_id"])
    record = {
        "trial_id": f"llm-{prefix}-{trial_seq:03d}",
        "arm": "llm",
        "axis": axis,
        "tier": int(payload.get("tier", 1)),
        "parent_trial": parent_trial,
        "action_id": action_id,
        "is_free_proposal": action_id.startswith("free-"),
        "hypothesis": str(payload["hypothesis"]),
        "evidence_anchor": str(payload["evidence_anchor"]),
        "target_slices": list(payload["target_slices"]),
        "expected_effect": str(payload.get("expected_effect", "")),
        "falsification": str(payload["falsification"]),
        "editable_paths": list(payload["editable_paths"]),
        "patch_plan": str(payload.get("patch_plan", "")),
        "budget": {"gpu_hours_cap": gpu_hours_cap, "seeds": [screening_seed]},
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if axis == "model":
        extra = payload.get("model_axis_extra")
        if not isinstance(extra, dict):
            return ParsedProposal(
                ok=False,
                errors=["model axis proposal must include model_axis_extra (param counts filled at preflight)"],
            )
        record["model_axis_extra"] = extra

    try:
        validate_trial(record)
    except ContractViolation as exc:
        return ParsedProposal(ok=False, errors=str(exc).splitlines())
    return ParsedProposal(ok=True, trial_record=record)
