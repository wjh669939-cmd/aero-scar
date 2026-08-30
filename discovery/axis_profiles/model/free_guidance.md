# M 轴自由提案边界（工程级，事实性约束；DEC-001 激活前本轴不可排批）

- 可编辑文件（每 trial 声明其一）：fusion/dual_stream_fusion.py、
  encoders/exogenous_encoder.py、encoders/frets_encoder.py、encoders/transformer_encoder.py；
  （unified_model.py 头部段属 tier2 函数拼接协议，不在本轴）
- 参数预算：总参数量相对 parent 实测 3,930,853 变化 |Δ| ≤ 5%（冒烟内机器断言，超限零 GPU 拒）；
- 实测张量流转（measured_interface_v1.json，0830 探针）：encoder_T/F 输入 (B·slots, 11, 96)、
  输出 (B·slots, 256, 87)；DualStreamFusion 融合两路 (B·slots, 256) → (B·slots, 256)；
- 实测事实：预训练前向不经过 exo_encoder（exo 仅在下游 encode 路径被消费）；
- 阶段绑定（v1.3 草案，待 C 签）：exo_encoder 编辑按 tier1 绑定并可复用 pretrain ckpt；
  其余编辑面按 tier2 绑定，预训练与四条下游腿全部重跑；
- 构造函数与公共方法签名冻结（调用方 unified_model.py 锁定不可编辑）；
- IO 形状/dtype、前后向有限性、双指纹行为惰性（pretrain 损失 + encode 含 exo 输出）均为机器闸门。

## 通用记账事实（全轴一致，来源：22 号方案 / decision_policy v1.2）

- 自由提案（action_id 以 free- 开头）须附 non_expressibility（≥30 字）说明为何模板+参数不可表达；
- 与活跃模板机制实质等价的"自由"提案会被机械拦截并要求重提；
- 与基线语义等价（仅注释/docstring 改动）的编辑会被 no-op 闸门拒绝，不进训练；
- 三类别分离记账：random / llm_template / llm_free；自由提案不进两臂对照，单列全量报告；
- 评价标准零特权：同一轴锁、同一冒烟闸门、同一筛选线、同一 3-seed 确认、同一隐藏认证。

（本文件只陈述边界与事实，不提供任何机制方向建议——机制假设必须由提案方自行形成，
2026-08-28 统筹纪律。）
