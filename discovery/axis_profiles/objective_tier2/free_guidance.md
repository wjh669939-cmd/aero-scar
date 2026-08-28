# O-tier2 轴自由提案边界（工程级，事实性约束）

- 编辑面为 unified_pretrain_forward 单个方法（函数级拼接协议，方法外机器保证不变）；
- soft_dtw_cuda.py 内核锁定：可改调用方式与权重，不可改内核本身；
- mask_ratio（遮挡比例）为冻结配置；random:causal 配比（causal_prob 实参）仅当 patch_plan
  明确声明时可改；
- 方法体内禁 import；可用模块级已导入的名字（torch/F/generate_hybrid_mask/apply_mask/
  masked_mse_loss 及 self.* 成员）；
- 裁决事实（decision_policy v1.2）：本轴主指标 = candidate forecast_pretrained 对
  parent forecast_scratch 的配对差；每 trial 全流程重跑预训练（无 checkpoint 复用）。

## 通用记账事实（全轴一致，来源：22 号方案 / decision_policy v1.2）

- 自由提案（action_id 以 free- 开头）须附 non_expressibility（≥30 字）说明为何模板+参数不可表达；
- 与活跃模板机制实质等价的"自由"提案会被机械拦截并要求重提；
- 与基线语义等价（仅注释/docstring 改动）的编辑会被 no-op 闸门拒绝，不进训练；
- 三类别分离记账：random / llm_template / llm_free；自由提案不进两臂对照，单列全量报告；
- 评价标准零特权：同一轴锁、同一冒烟闸门、同一筛选线、同一 3-seed 确认、同一隐藏认证。

（本文件只陈述边界与事实，不提供任何机制方向建议——机制假设必须由提案方自行形成，
2026-08-28 统筹纪律。）
