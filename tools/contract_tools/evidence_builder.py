"""A-9: 机器生成的证据表（替代手写失败切片摘要）。

动机（8/27 复盘）：LLM 的假设质量受限于喂给它的证据；手写摘要带转述偏差且不随
实验进化。本模块从三处自动生成证据文本：
1. campaign 静态事实（三 seed 方差报告结论，冻结不变）；
2. parent 的全网格评测指标（C evaluator 落盘的 metrics.json，机器排名最弱切片）；
3. lineage 摘要（已试过的机制与结局，由调用方拼接 context_assembler 已有逻辑）。

产出文本进提示词前必须过 assert_no_hidden_tokens（调用方负责，本模块只用 val 指标）。
"""

from __future__ import annotations

import json
from pathlib import Path

# 三 seed 方差报告（D1，2026-08-27）的冻结结论——不随单 seed 指标变化
STATIC_CAMPAIGN_FACTS = (
    "(static) 预训练对 forecast 是负迁移：3 seed（42/43/2027）scratch RMSE 一致优于 "
    "pretrained，配对 SNR 3.48，三 seed 同号；对 classification 为正迁移但幅度不稳"
    "（macro_f1 增益 +0.077/+0.007 波动）。"
    "(static) 主指标跨 seed 噪声底 0.00023；筛选线 = 配对改善 >= 0.0005。"
    "(static) hazard 类 F1 跨 seed 波动大（0.53~0.79），val 中 hazard 支撑仅 ZBAA"
    "（support=94），ZSPD/ZSSS val 标签近单类（degenerate）。"
    "(static) 输入侧风以 (wind_x, wind_y) 归一化分量表示；ZSSS T+1 wind_x 存在回归头"
    "微溢出（>1 的预测值）。"
)


def _load_metrics(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _endpoint_map(metrics: dict) -> dict[str, dict]:
    return {
        ep["name"]: ep
        for ep in metrics.get("endpoints", [])
        if isinstance(ep, dict) and "name" in ep
    }


def forecast_slice_table(scratch: dict, pretrained: dict | None, top_n: int = 5) -> str:
    """最弱预测切片排名 + 分时距宏平均 + 负迁移逐点统计。"""
    eps = _endpoint_map(scratch)
    rmse = {n: e for n, e in eps.items() if n.endswith(".rmse") and not e.get("degenerate")}
    if not rmse:
        return ""
    lines = []
    worst = sorted(rmse.items(), key=lambda kv: kv[1]["value"], reverse=True)[:top_n]
    lines.append(
        "最弱预测切片 (scratch RMSE, 降序): "
        + "; ".join(f"{n.removeprefix('forecast.')}={e['value']:.4f}" for n, e in worst)
    )
    by_h: dict[str, list[float]] = {}
    for n, e in rmse.items():
        for h in ("T+1", "T+4", "T+8"):
            if f".{h}." in n:
                by_h.setdefault(h, []).append(e["value"])
    if by_h:
        lines.append(
            "分时距 RMSE 宏平均: "
            + ", ".join(f"{h}={sum(v) / len(v):.4f}" for h, v in sorted(by_h.items()))
        )
    if pretrained:
        pre = _endpoint_map(pretrained)
        worse = [
            n for n in rmse
            if n in pre and pre[n]["value"] > rmse[n]["value"]
        ]
        if rmse:
            lines.append(
                f"负迁移逐点统计: pretrained 劣于 scratch 的 RMSE 切片 "
                f"{len(worse)}/{len(rmse)} 个"
            )
    return "\n".join(lines)


def classification_slice_table(scratch: dict, pretrained: dict | None) -> str:
    eps = _endpoint_map(scratch)
    f1 = {n: e for n, e in eps.items() if ".f1_" in n}
    if not f1:
        return ""
    valid = sorted(
        ((n, e) for n, e in f1.items() if not e.get("degenerate")),
        key=lambda kv: kv[1]["value"],
    )
    degen = [n.removeprefix("classification.") for n, e in f1.items() if e.get("degenerate")]
    lines = []
    if valid:
        lines.append(
            "最弱分类切片 (scratch F1, 升序): "
            + "; ".join(f"{n.removeprefix('classification.')}={e['value']:.3f}" for n, e in valid[:4])
        )
    if degen:
        lines.append(f"degenerate 切片（不计入 overall）: {', '.join(degen[:8])}")
    if pretrained:
        pol_s = scratch.get("decision_policy_metrics", {})
        pol_p = pretrained.get("decision_policy_metrics", {})
        pairs = []
        for key in ("classification_macro_f1", "hazard_class_f1"):
            if key in pol_s and key in pol_p:
                pairs.append(f"{key}: scratch {pol_s[key]['value']:.3f} vs pretrained {pol_p[key]['value']:.3f}")
        if pairs:
            lines.append("预训练增益参考: " + "; ".join(pairs))
    return "\n".join(lines)


def build_failure_slices(parent_refs_dir: Path, static_facts: str = STATIC_CAMPAIGN_FACTS,
                         max_chars: int = 2600) -> str:
    """组装完整证据文本。parent_refs_dir 下按目录名寻找四条腿的 metrics.json。"""
    legs = {
        "forecast_scratch": None,
        "forecast_pretrained": None,
        "classification_scratch": None,
        "classification_pretrained": None,
    }
    for sub in parent_refs_dir.iterdir() if parent_refs_dir.is_dir() else []:
        for leg in legs:
            # 目录命名兼容: seed42_forecast_scratch / seed42_cls_scratch 等
            key = leg.replace("classification", "cls")
            if leg in sub.name or key in sub.name:
                legs[leg] = _load_metrics(sub / "out" / "metrics.json")
    blocks = [static_facts]
    fc = forecast_slice_table(legs["forecast_scratch"] or {}, legs["forecast_pretrained"])
    if fc:
        blocks.append("[auto] parent(seed42) 预测网格：\n" + fc)
    cls = classification_slice_table(
        legs["classification_scratch"] or {}, legs["classification_pretrained"]
    )
    if cls:
        blocks.append("[auto] parent(seed42) 分类网格：\n" + cls)
    text = "\n\n".join(blocks)
    return text[:max_chars]
