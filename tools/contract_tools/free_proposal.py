"""A-10：自由提案触发与校验（22 号方案落地，2026-08-27 预注册）。

规则（先于任何候选结果冻结）：
- 强制触发：同轴连续 CONSECUTIVE_REFUTED_THRESHOLD 个 trial 被裁决 refuted，
  且该轴 active 模板已全部尝试过（豁免条款），则下一发必须自由提案；
- 真实性：自由提案必须携带 non_expressibility（>= MIN_NON_EXPRESSIBILITY_CHARS 字符），
  说明机制为何不能被现有模板 + 参数表达；
- 记账：自由提案不进入 LLM-vs-随机 两臂对照，单列报告（见 decision_policy 备注）；
- 平等：评价标准与模板提案完全一致，无特权。
"""

from __future__ import annotations

CONSECUTIVE_REFUTED_THRESHOLD = 2
MIN_NON_EXPRESSIBILITY_CHARS = 30

FORCED_FREE_DIRECTIVE = (
    "\n\nMANDATORY THIS ROUND: the template actions on this axis have been "
    "exhausted and consecutively refuted. You MUST submit a FREE proposal "
    "(action_id starting with 'free-') implementing a mechanism outside the "
    "template space, and you MUST include a 'non_expressibility' field "
    "(>= 30 chars) explaining why no existing template + parameters can "
    "express this mechanism. Template action_ids will be rejected this round."
)


def trial_history(lineage_records: list[dict]) -> list[tuple[str, str, str, str]]:
    """按时间序提取 (trial_id, axis, action_id, verdict)；verdict_backfill 事件可覆盖。"""
    verdict: dict[str, str] = {}
    order: list[tuple[str, str, str]] = []
    for rec in lineage_records:
        event = rec.get("event")
        if event == "trial_done":
            order.append((rec.get("trial_id", ""), rec.get("axis", ""), rec.get("action_id", "")))
            verdict[rec.get("trial_id", "")] = rec.get("hypothesis_verdict") or "not_evaluated"
        elif event == "verdict_backfill" and rec.get("trial_id"):
            verdict[rec["trial_id"]] = rec.get("hypothesis_verdict") or "not_evaluated"
    return [(t, a, act, verdict.get(t, "not_evaluated")) for t, a, act in order]


def forced_free_status(
    lineage_records: list[dict],
    axis: str,
    active_template_ids: list[str],
) -> tuple[bool, str]:
    """判定该轴下一发是否强制自由提案。返回 (是否强制, 依据说明)。"""
    rows = [r for r in trial_history(lineage_records) if r[1] == axis]
    trailing_refuted = 0
    for _, _, _, v in reversed(rows):
        if v == "refuted":
            trailing_refuted += 1
        else:
            break
    if trailing_refuted < CONSECUTIVE_REFUTED_THRESHOLD:
        return False, f"连续 refuted {trailing_refuted} < 阈值 {CONSECUTIVE_REFUTED_THRESHOLD}"
    tried = {act for _, _, act, _ in rows if act}
    untried = sorted(set(active_template_ids) - tried)
    if untried:
        return False, f"豁免：该轴仍有未试模板 {untried}（顺延一发）"
    return True, f"连续 refuted {trailing_refuted} 且 active 模板已穷举，强制自由提案"


def validate_free_proposal(record: dict) -> list[str]:
    """自由提案的真实性校验。返回错误列表（空 = 通过）。"""
    errors: list[str] = []
    if not record.get("is_free_proposal"):
        return errors
    text = str(record.get("non_expressibility", "")).strip()
    if len(text) < MIN_NON_EXPRESSIBILITY_CHARS:
        errors.append(
            f"自由提案必须携带 non_expressibility（>= {MIN_NON_EXPRESSIBILITY_CHARS} 字符，"
            f"当前 {len(text)}）：说明机制为何不能被现有模板+参数表达"
        )
    return errors
