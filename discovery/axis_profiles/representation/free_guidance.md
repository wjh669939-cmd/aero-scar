# R 轴自由提案边界（工程级，事实性约束）

- 可编辑文件仅 trial_features.py；
- 硬红线（泄漏）：FORBIDDEN_INPUT_COLUMNS = ("weather_code_id", "weather_label", "significant_wx")
  不得以任何形式进入输入；任何特征只能使用 forecast_origin 时刻及之前可得的信息（issue-time 因果）；
- build_forecast_inputs / build_classification_inputs 的输出 x 形状锁定为 (96, n_slots, 2)，
  node_mask 为 bool；
- 事实（lineage 实证）：预报分支只消费 x 张量本身；exo/编码器路径的改动不影响 forecast 两腿。

## 通用记账事实（全轴一致，来源：22 号方案 / decision_policy v1.2）

- 自由提案（action_id 以 free- 开头）须附 non_expressibility（≥30 字）说明为何模板+参数不可表达；
- 与活跃模板机制实质等价的"自由"提案会被机械拦截并要求重提；
- 与基线语义等价（仅注释/docstring 改动）的编辑会被 no-op 闸门拒绝，不进训练；
- 三类别分离记账：random / llm_template / llm_free；自由提案不进两臂对照，单列全量报告；
- 评价标准零特权：同一轴锁、同一冒烟闸门、同一筛选线、同一 3-seed 确认、同一隐藏认证。

（本文件只陈述边界与事实，不提供任何机制方向建议——机制假设必须由提案方自行形成，
2026-08-28 统筹纪律。）
