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
