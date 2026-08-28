# decision_policy v1.1 签字冻结台账

> 本台账独立于被签文件，被签正文的 SHA-256 永久可验。签字后对
> `decision_policy_v1_draft.json` 的任何修改都构成版本变更，须走修订流程并重新签字。

## 被签文本

- 文件：`00_contract/decision_policy_v1_draft.json`
- SHA-256：`eb49492b85b3671ce31f36ab6197c5bb0ccf83a602093c1d1803ac4ed3cfb9dd`
- 内容要点：primary = RMSE_macro_norm（筛选 ≥0.0005 / 确认 ≥0.0010 + 三 seed 同号 + CI 不含零）；
  分类三指标作硬 guardrail；hazard 禁作选优；imputation_grid_mean = C1 `overall.imputation.mse`
  （软 guardrail，阈值待首批含插补 trial 后标定）；三类别记账口径（random / llm_template / llm_free）。
- 双机哈希对账：本机 = D 机 = 签字哈希，一致（2026-08-27 23:00 验证）。

## 签字记录

| 角色 | 结论 | 日期 |
|---|---|---|
| C（评测侧 / test custodian） | 已审阅，同意按当前文本冻结，**不再因候选结果调整规则或数值** | 2026-08-27 |
| A（框架/统筹） | 确认哈希对账一致，入库本台账 | 2026-08-27 |
| 导师 | 按 18 文档原则（技术决策组内自决）仅作告知，不构成生效条件 | 告知于站会 |

## 生效语义

- **状态：FROZEN（2026-08-27 23:00，C + A 双签生效）**——早于任何候选结果；
- discovery 筛选、确认、候选冻结全部按本文本执行；
- 唯一修订通道：任何候选确认之前，经 C + A 重签；此后本 campaign 按已签文本执行到底。


---

## v1.2 重签条目（2026-08-28 发起，待 C 签字）

- 文件：00_contract/decision_policy_v1.2_draft.json
- SHA-256：7268682a2c31dc9e734ef1e95bf1922d2a4f5f8bcbbe0c15ead91a9c4aa08f71
- 变更范围：仅新增 primary_endpoint.stage_binding（默认轴 = forecast_scratch，与 v1.1 实现一致；
  objective_tier2 = forecast_pretrained vs parent forecast_scratch）。阈值、guardrail、生命周期、
  反作弊条款零改动。逐行 diff：00_contract/decision_policy_v1.1_to_v1.2.diff（56 行）。
- 触发与依据：llm-obj-007 暴露 tier2 主指标恒零缺口（30 号文档）；修订发生于任何 screen_pass
  之前，走 v1.1 预留通道（候选确认前 C+A 重签）。
- 生效语义：C 签字前 v1.1 继续执行（tier2 trial 照跑，证据完整落盘）；签字后 A 更新驱动
  adjudicate() 并对 v1.1 时代的 tier2 trial（llm-obj-007、llm-obj-010）出 verdict_backfill
  双版本对照记录。

| 角色 | 结论 | 日期 |
|---|---|---|
| A（框架/统筹） | 起草并确认 v1.1 原文哈希未动（eb49492b…） | 2026-08-28 |
| C（评测侧 / test custodian） | **已同意按 SHA 7268682a… 冻结**（经统筹转达） | 2026-08-28 21:00 |

**状态：v1.2 FROZEN（2026-08-28 21:00，C + A 双签生效）**——先于任何 screen_pass。
被签文件字节不动（与 v1.1 惯例一致，台账为生效状态的唯一真相源）。
生效动作：驱动 adjudicate() v1.2 tier2 分支上线（7 例单测过）；llm-obj-007 / llm-obj-010
verdict_backfill 双版本对照见 lineage。
