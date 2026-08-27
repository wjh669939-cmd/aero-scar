"""G-5：lineage → LLM 提案上下文组装器。

职责：
- 从 action_registry 取当前轴 active 模板、从 lineage 记录取最近 N 条摘要、
  从 failure_slices 报告取可见验证集切片，填入 prompts/llm_proposal.md；
- 嵌入 INTERFACE CONTRACT（G-8 抽薄后 trial 可写文件与保持签名）；
- 组装完成后强制过隐藏 token 检查（ZBAD/EVALUATION/sealed/test 指标等），
  命中即抛异常——宁可停机也不把认证信息喂给 LLM。
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPT_TEMPLATE = Path(__file__).parent / "prompts" / "llm_proposal.md"

# 与 axis_lock hidden_data_guard 同源；此处另加 test 指标关键词
HIDDEN_TOKENS = (
    "AeroWF_v1_EVALUATION",
    "sealed/",
    "ZBAD",
    "pretrain/test",
    "test_metrics",
    "certification_result",
    "c_evaluator_private",
)

INTERFACE_CONTRACTS = {
    "representation": (
        "- editable file: aerowf_downstream_v2/src/trial_features.py ONLY;\n"
        "- must keep signatures: build_forecast_inputs(...), "
        "AllowedContextEncoder(nn.Module), build_classification_inputs(...);\n"
        "- forbidden input columns (asserted by the locked caller): "
        "weather_code_id, weather_label, significant_wx;\n"
        "- target generation (what to predict) lives in locked files; you may "
        "only change how inputs are represented."
    ),
    "objective_tier1": (
        "- editable file: aerowf_downstream_v2/src/trial_objective.py ONLY;\n"
        "- must keep signatures: forecast_loss(prediction, target, node_mask), "
        "classification_loss(logits, label, *, class_weights), "
        "compute_class_weights(train_label_counts);\n"
        "- class statistics are computed from the train split by the locked "
        "caller; you may change how weights/losses are formed, not which "
        "split feeds them;\n"
        "- losses must stay differentiable and finite on masked/virtual runways."
    ),
    "objective_tier2": (
        "- editable files: models/AirFM/unified_model.py (lambda assembly in "
        "unified_pretrain_forward only), physics_distance.py, masked.py;\n"
        "- soft_dtw_cuda.py is locked: change its invocation/weights, not the kernel;\n"
        "- pretraining data manifest and mask ratio protocol are frozen."
    ),
    "model": (
        "- editable paths: models/AirFM/fusion/**, models/AirFM/encoders/**, "
        "UnifiedSeries2Vec.encode assembly in unified_model.py;\n"
        "- parameter budget cap enforced by validator (param_budget_counter);\n"
        "- I/O tensor shapes must be preserved (io_shape_check)."
    ),
}


class HiddenInfoLeak(RuntimeError):
    """组装出的上下文包含认证/测试相关 token。"""


def assert_no_hidden_tokens(text: str) -> None:
    hits = [tok for tok in HIDDEN_TOKENS if tok in text]
    if hits:
        raise HiddenInfoLeak(f"prompt context contains hidden tokens: {hits}")


def summarize_lineage(records: list[dict], max_records: int = 10) -> str:
    """result 记录 → 每条一行的谱系摘要（只取可见字段，不透传原始 metrics 全文）。"""
    lines = []
    for rec in records[-max_records:]:
        primary = rec.get("primary_metric", {})
        lines.append(
            "- trial={id} axis={axis} action={action} verdict={verdict} "
            "primary[{name}]={value} note={note}".format(
                id=rec.get("trial_id", "?"),
                axis=rec.get("axis", "?"),
                action=rec.get("action_id", "?"),
                verdict=rec.get("verdict", "pending"),
                name=primary.get("name", "RMSE_macro_norm"),
                value=primary.get("value", "n/a"),
                note=(rec.get("verdict_note") or "")[:120],
            )
        )
    return "\n".join(lines) if lines else "- (no prior trials in this campaign)"


def active_templates(registry_path: Path, axis: str) -> str:
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    picked = []
    for action in reg.get("actions", []):
        if action.get("axis") == axis and action.get("status") == "active":
            picked.append(
                {
                    k: action.get(k)
                    for k in (
                        "action_id",
                        "tier",
                        "type",
                        "evidence_anchor",
                        "hypothesis",
                        "target_slices",
                        "param_space",
                    )
                    if action.get(k) is not None
                }
            )
    return json.dumps(picked, ensure_ascii=False, indent=2) if picked else "[]"


def assemble_proposal_prompt(
    axis: str,
    registry_path: Path,
    lineage_records: list[dict],
    failure_slices_summary: str,
    max_lineage: int = 10,
) -> str:
    if axis not in INTERFACE_CONTRACTS:
        raise ValueError(f"unknown axis for prompt assembly: {axis}")
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    body = template.split("---", 1)[1] if "---" in template else template
    prompt = (
        body.replace("{AXIS}", axis)
        .replace("{INTERFACE_CONTRACT}", INTERFACE_CONTRACTS[axis])
        .replace("{FAILURE_SLICES_SUMMARY}", failure_slices_summary.strip())
        .replace("{ACTIVE_TEMPLATES}", active_templates(registry_path, axis))
        .replace("{LINEAGE_SUMMARY}", summarize_lineage(lineage_records, max_lineage))
        .strip()
    )
    assert_no_hidden_tokens(prompt)
    return prompt
