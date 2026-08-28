# O-tier1 轴自由提案边界（工程级，事实性约束）

- 可编辑文件仅 trial_objective.py；三个函数签名必须保持（见 interface.md）；
- 张量布局（事实）：prediction/target 为 (batch, runway_slots, horizons, components)，
  dim=2 才是时距 [T+1,T+4,T+8]，dim=3 为 [wind_x, wind_y]；
- 数值域（事实）：目标为 [0,1] 归一化尺度；损失须对该尺度有意义、可微、在虚拟槽位掩码下有限；
- O2 模板的事件加权依赖 B3 交付的 is_hazard_event_T1/T4/T8 布尔列（交付前事件口径不可自造代理）。

## 通用记账事实（全轴一致，来源：22 号方案 / decision_policy v1.2）

- 自由提案（action_id 以 free- 开头）须附 non_expressibility（≥30 字）说明为何模板+参数不可表达；
- 与活跃模板机制实质等价的"自由"提案会被机械拦截并要求重提；
- 与基线语义等价（仅注释/docstring 改动）的编辑会被 no-op 闸门拒绝，不进训练；
- 三类别分离记账：random / llm_template / llm_free；自由提案不进两臂对照，单列全量报告；
- 评价标准零特权：同一轴锁、同一冒烟闸门、同一筛选线、同一 3-seed 确认、同一隐藏认证。

（本文件只陈述边界与事实，不提供任何机制方向建议——机制假设必须由提案方自行形成，
2026-08-28 统筹纪律。）
