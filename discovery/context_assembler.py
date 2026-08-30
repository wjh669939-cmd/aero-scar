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
        "only change how inputs are represented;\n"
        "- TENSOR LAYOUT (factual, verified on the frozen release): runway_arr and the "
        "returned x are (n_slots, 96, 11) = (runway slots, time steps, channels); "
        "channel index 1 = wind_x, index 2 = wind_y (the forecast targets); the "
        "downstream model input layer expects exactly 11 channels — appending or "
        "removing channels is structurally impossible; transformations must stay "
        "within the (n_slots, 96, 11) layout;\n"
        "- both builders accept keyword runway_axis_heading_deg: an (n_slots,) float32 "
        "array of nominal runway axis headings in degrees mod 180 (frozen B3 delivery; "
        "slot-to-runway mapping carries a documented provenance limitation), with NaN "
        "for virtual/unknown slots. The baseline ignores it; any use must tolerate NaN "
        "and must not inject NaN into x."
    ),
    "objective_tier1": (
        "- editable file: aerowf_downstream_v2/src/trial_objective.py ONLY;\n"
        "- must keep signatures: forecast_loss(prediction, target, node_mask), "
        "classification_loss(logits, label, *, class_weights), "
        "compute_class_weights(train_label_counts);\n"
        "- class statistics are computed from the train split by the locked "
        "caller; you may change how weights/losses are formed, not which "
        "split feeds them;\n"
        "- losses must stay differentiable and finite on masked/virtual runways.\n"
        "- TENSOR LAYOUT (factual, from the locked caller): prediction/target are "
        "(batch, runway_slots, horizons, components) = (B, 4 or 5, 3, 2); dim=1 is "
        "RUNWAY SLOTS, dim=2 is horizons [T+1, T+4, T+8], dim=3 is [wind_x, wind_y]; "
        "node_mask is (batch, runway_slots). Horizon-wise weights must be applied on "
        "dim=2, e.g. weights.view(1, 1, -1, 1)."
    ),
    "objective_tier2": (
        "- editable files: models/AirFM/unified_model.py (lambda assembly in "
        "unified_pretrain_forward only), physics_distance.py, masked.py;\n"
        "- soft_dtw_cuda.py is locked: change its invocation/weights, not the kernel;\n"
        "- pretraining data manifest and mask ratio protocol are frozen."
    ),
    "model": (
        # 全部为 0830 探针实测（measured_interface_v1.json），非推测
        "- editable files (declare exactly ONE in editable_paths): "
        "models/AirFM/fusion/dual_stream_fusion.py, "
        "models/AirFM/encoders/exogenous_encoder.py, "
        "models/AirFM/encoders/frets_encoder.py, "
        "models/AirFM/encoders/transformer_encoder.py;\n"
        "- measured tensor flow (real config): encoder_T and encoder_F each receive "
        "(batch*slots, 11, 96) and return (batch*slots, 256, 87); "
        "DualStreamFusion.forward(rep_T, rep_F) fuses two (batch*slots, 256) vectors "
        "into one (batch*slots, 256); ExogenousEncoder.forward(exo_categorical, "
        "exo_continuous) consumes 4 categorical ids + 3 continuous scalars per sample;\n"
        "- IMPORTANT: the pretraining forward pass does NOT execute ExogenousEncoder "
        "(exogenous inputs are only consumed on the downstream encode path);\n"
        "- total parameter count must stay within +/-5% of baseline 3,930,853 "
        "(mechanically enforced before training);\n"
        "- constructor signatures and all public method signatures/shapes are frozen "
        "(callers in unified_model.py are locked);\n"
        "- I/O tensor shapes and dtypes must be preserved."
    ),
}


class HiddenInfoLeak(RuntimeError):
    """组装出的上下文包含认证/测试相关 token。"""


def assert_no_hidden_tokens(text: str) -> None:
    hits = [tok for tok in HIDDEN_TOKENS if tok in text]
    if hits:
        raise HiddenInfoLeak(f"prompt context contains hidden tokens: {hits}")


_DELTA_LABELS = (
    ("forecast_scratch.RMSE_macro_norm", "fc_scratch RMSE"),
    ("forecast_scratch.MAE_macro_norm", "fc_scratch MAE"),
    ("forecast_pretrained.RMSE_macro_norm", "fc_pretrained RMSE"),
    ("tier2_primary_pretrained_vs_parent_scratch.RMSE_macro_norm", "tier2_primary(pre_vs_parent_scratch) RMSE"),
    ("classification_scratch.classification_macro_f1", "cls_scratch macroF1"),
    ("classification_pretrained.classification_macro_f1", "cls_pretrained macroF1"),
    ("classification_pretrained.hazard_class_f1", "cls_pretrained hazardF1"),
)


def _delta_table(deltas: dict) -> str:
    """配对 Δ 的对齐单行表（负值 = 改善，RMSE/MAE 类；正值 = 改善，F1 类）。"""
    cells = []
    for key, label in _DELTA_LABELS:
        value = deltas.get(key)
        if isinstance(value, (int, float)):
            cells.append(f"{label} {value:+.6f}")
    return " | ".join(cells) if cells else "(no paired deltas)"


def summarize_lineage(records: list[dict], max_records: int = 10) -> str:
    """lineage 记录 → 谱系摘要（如实呈现裁决/配对 Δ/审计注记；不透传原始 metrics 全文）。

    2026-08-29 修复：旧版读取的字段名（verdict/primary_metric）与 runner 实际写入的
    记录（hypothesis_verdict/paired_delta_vs_parent）不匹配，导致此前提示词中的谱系
    摘要恒为 pending/n/a——LLM 从未看到过任何配对结果。本版按真实记录结构渲染。
    """
    lines: list[str] = []
    for rec in records[-max_records:]:
        event = rec.get("event")
        tid = rec.get("trial_id", "?")
        if event == "trial_done":
            verdict = rec.get("hypothesis_verdict") or "pending"
            lines.append(
                f"- trial={tid} axis={rec.get('axis', '?')} action={rec.get('action_id', '?')} "
                f"arm={rec.get('arm_category', '?')} status={rec.get('status', '?')} verdict={verdict}"
            )
            basis = (rec.get("verdict_basis") or "")[:160]
            if basis:
                lines.append(f"    basis: {basis}")
            deltas = rec.get("paired_delta_vs_parent") or {}
            lines.append(f"    paired-delta vs parent: {_delta_table(deltas)}")
            hyp = (rec.get("hypothesis") or "")[:160]
            if hyp:
                lines.append(f"    hypothesis: {hyp}")
        elif event == "verdict_backfill":
            lines.append(
                f"- verdict update: trial={tid} -> {rec.get('verdict', '?')} "
                f"(policy {rec.get('policy_version', 'v1')}): {(rec.get('verdict_basis') or '')[:140]}"
            )
        elif event == "audit_note":
            lines.append(f"- AUDIT NOTE trial={tid}: {(rec.get('note') or '')[:300]}")
        elif event in ("smoke_rejected", "proposal_rejected", "codegen_rejected"):
            reason = (rec.get("smoke_error") or "; ".join(rec.get("errors") or []) or "")[:140]
            lines.append(f"- gate-rejected ({event}) trial={tid} axis={rec.get('axis', '?')}: {reason}")
        elif event == "driver_error":
            lines.append(f"- driver_error axis={rec.get('axis', '?')}: {(rec.get('error') or '')[:100]}")
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
    max_lineage: int = 18,
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
