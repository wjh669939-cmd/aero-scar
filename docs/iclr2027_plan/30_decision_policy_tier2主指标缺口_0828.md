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

- [x] v1.2 草案已起草（2026-08-28 17:50）：`00_contract/decision_policy_v1.2_draft.json`，
      SHA-256 `7268682a…`；逐行 diff `decision_policy_v1.1_to_v1.2.diff`（56 行）；
      SIGNOFF 台账已追加待签条目；v1.1 原件未动（`eb49492b…` 校验一致）；
- [ ] **C 审阅 + 回签**（今晚）：变更只有 stage 绑定一节，阈值零改动，评测器零改动、
      无需重新交付任何东西；
- [ ] 重签后 A 改 `adjudicate()`（tier2 分支）+ 对 obj-007 / obj-010
      按 v1.2 出**补充裁决记录**（verdict_backfill，注明双版本裁决一致/不一致）；
- [x] 重签前 tier2 trial 照常跑（证据仍完整落盘），lineage 消费方知悉标签语义缺口。

## 六、同类问题排查（2026-08-28 全轴自查）

对照「轴的编辑面 × 裁决主指标的响应面」逐轴过了一遍：

| 轴 | 编辑面能否触达 forecast_scratch（现行主指标） | 结论 |
|---|---|---|
| representation | **仅当变换 `x` 风分量本身**（rep-002/006 实证：exo/编码器路径 Δ 精确为 0） | 非缺口，但攻击面窄——排批权重议题（另行讨论） |
| objective_tier1 | ✅ 直接改下游损失，scratch 腿响应（obj-008 实证 -0.000451） | 无问题 |
| objective_tier2 | ❌ 结构性恒零 | **本文档主题，v1.2 修复** |
| model（条件轴，未激活） | ✅ 结构改动影响全部腿 | 激活时无此缺口 |

**遗留发现（不进 v1.2，站会议）**：

1. **O3（分类 focal 模板）与主指标结构性脱钩**——它只改 classification_loss，
   forecast 两腿必然 Δ=0，在任何版本政策下都不可能 screen_pass。这不是 bug
   （政策有意把分类降为 guardrail），但意味着 O3 作为"候选产生器"是死路，
   只能产出 guardrail/机制证据。选项：(a) 接受其证据价值、排批降权；
   (b) 停用省 GPU；(c) 另立分类主指标赛道（动作大，需导师层面）。
2. imputation_grid_mean 软 guardrail 仍待 C 加映射键 + 阈值标定（已知，非阻塞，
   当前 trial 不产出插补腿）。
3. 确认阶段数据齐备性核过：tier2 按 v1.2 需要各 seed 的 parent scratch 参考，
   五 seed 表已全（42/43/2027/3407/5519），无阻塞。

---

*A 侧记录。obj-007 产物：`discovery/trials/llm-obj-007/`（含 tier2_function_diff.txt）。*
