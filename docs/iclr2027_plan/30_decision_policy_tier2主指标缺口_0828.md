# decision_policy v1.1 的 tier2 主指标缺口（发现于 llm-obj-007）

> 2026-08-28 14:36 记录。性质：**冻结政策的语义缺口**，修订走 v1.1 预留通道
> （任何候选确认之前，C + A 重签）。在重签前，驱动裁决代码**不做任何单方修改**。

## 一、现象

llm-obj-007（O5，objective_tier2 首发）completed，配对结果：

| 腿 | Δ vs parent |
|---|---|
| forecast_scratch RMSE | **0.000000（bit 级不变）** |
| forecast_pretrained RMSE | **+0.003152（恶化）** |
| classification_pretrained macro_f1 | +0.011 |
| classification_pretrained hazard_f1 | -0.049 |

裁决：refuted（verdict_rule_v1：主指标 forecast_scratch Δ 未达筛选线）。

## 二、结构性问题

**tier2 只改预训练目标 → scratch 两腿与 parent bit 级相同 → 主指标
（forecast_scratch.RMSE）对 tier2 永远 Δ=0 → 一切 tier2 trial 自动 refuted，
与其编辑质量无关。**

verdict_rule_v1 的主指标锚在 scratch 腿，而 tier2 的注册证伪条件（O4-O6）
锚在 pretrained-vs-scratch 关系（"T+1 仍落后 Scratch，或 T+8/插补优势丢失"）。
两者错位：政策测不到 tier2 的目标效应。

本次无误判之实害：obj-007 的 pretrained 腿实际**恶化**（+0.0032），按其注册
证伪条件同样 refuted——标签碰巧正确，但依据文本（"无可测效应"）语义错误。

## 三、风险（重签前）

1. 今晚 batch05 的 trial 10（tier2）同样会被标 refuted，**即使其 pretrained 腿
   大幅改善**——错误标签会进入 lineage 并污染后续提案的证据上下文（LLM 可从
   配对 Δ 明细里自行辨认，但裁决标签的权重更高）；
2. tier2 连续两个 refuted 会触发该轴强制自由（22 文档规则 1）——若 refuted
   标签本身失真，触发也失真（豁免条款可挡一段：O4/O6 未试）。

## 四、修订提案（供 C+A 重签为 v1.2）

**tier2 专用主指标映射**（其余轴不变）：

- 筛选（单 seed）：`forecast_pretrained.RMSE_macro_norm` 相对 **parent
  forecast_scratch 参考值**（0.048471，seed42）配对改善 ≥ 0.0005 判 screen_pass
  ——语义 = "修复负迁移到反超裸训"，与 DEC-002 的激活理由一字不差；
- 兜底观测：`forecast_pretrained` vs parent `forecast_pretrained`（0.050906）
  的改善作次级证据记录（负迁移收窄但未反超 → 记 partial_effect，不过筛）；
- 确认（3 seed）与 guardrail 全部沿用 v1.1 数值，不动阈值；
- 裁决文本区分三态：refuted_on_pretrained（pretrained 腿恶化或无改善）/
  partial_effect / screen_pass。

理由：阈值零改动（0.0005/0.0010 沿用），只补一个 stage 映射；修订发生在
任何 screen_pass 之前，预注册干净性可辩护。

## 五、行动清单

- [ ] 站会 5 分钟过本提案；C + A 重签 → decision_policy v1.2 + 新 SHA 入 SIGNOFF 台账；
- [ ] 重签后 A 改 `adjudicate()`（tier2 分支）+ 对 obj-007（及可能的 trial 10）
      按 v1.2 出**补充裁决记录**（verdict_backfill，注明双版本裁决一致/不一致）；
- [ ] 重签前 tier2 trial 照常跑（证据仍完整落盘），lineage 消费方知悉标签语义缺口。

---

*A 侧记录。obj-007 产物：`discovery/trials/llm-obj-007/`（含 tier2_function_diff.txt）。*
