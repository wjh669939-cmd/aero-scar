# 完成清单总览（8/26 深夜盘点）

> 目录用途：各侧一份"已完成 ✅ / 进行中 🔶 / 未开始 ⬜"的完整清单 + discovery 开跑门槛总表。
> 状态基准日：2026-08-26 深夜。此后各侧清单由 A 随交付受理更新，只追加不覆盖。
> 配套：任务书 06/07/08，受理报告 13，CLH 审计 10，合同区 `workspace/runs/aerowf-v1/00_contract/`。

## Discovery 开跑门槛总表（缺一不可）

| # | 门槛 | 负责 | 状态 | 预计 |
|---|---|---|---|---|
| G-1 | 冻结数据交付 + 完整性核验 | B | ✅ 8/26 完成（194/194 哈希过） | — |
| G-2 | 任务协议（contract v2）生效 | B | ✅ 已入库 00_contract | — |
| G-3 | C1 evaluator v1.0 冻结 | C | ✅ 8/26 深夜（提前于 8/29） | — |
| G-4 | C2 认证封存（v3 补正） | C | ✅ 哈希链闭合 | — |
| G-5 | CLH P0 子进程隔离 + 轴配置对齐 | A | ✅ 8/26（19 项测试全绿） | — |
| G-6 | 契约工具链（schema/axis_lock/随机臂/解析器/evaluator 客户端） | A | ✅ 28 项单测全绿 | — |
| G-7 | **G1 执行器接线**：CLH trial → `aerowf_full_pipeline_v2.py` 子进程，成功判定按 D 手册 §6 六条件 | A+D | ✅ 8/27 凌晨真机验收：seed1001 1-epoch smoke 在 D 机全流程 success（693.8s，六条件全过，产物 `results/harness/trial_20260827_003243_seed1001_c71c23`） | — |
| G-8 | **文件抽薄改造**：R/O 轴对象抽为 trial_features.py / trial_objective.py；抽薄后 seed43 复跑一致性验证 | D+A | ✅ 8/27 验收关闭：seed43 复跑五阶段 delta 全 0（bit 级一致）；axis_lock 升 FROZEN | — |
| G-9 | **部署**：CLH+contract_tools+C 私有配置 → D 的 3090 机器 | A+C | ✅ 8/27 关闭：C 私有配置部署 + golden 自检 completed；`c_evaluator_private` 已入禁访 token（axis_lock + LLM 上下文双拦截） | — |
| G-10 | **真实联调一轮**：真 checkpoint → predictions.npz → C evaluator → 全网格落盘 | A+C+D | ✅ 8/27 关闭：classification/forecast 双腿 completed（15+36 endpoint 全网格 + CI），manifest 哈希对账一致；新增 predictions_adapter + evaluator_client 按冻结版重写（55 测全绿）；**两个接口缺口待修**（19 文档：D 全覆盖导出 + C 值域范围决策） | — |
| G-11 | **D1 方差报告**（3 seed：42/43/2027，冻结数据）→ decision_policy 数值标定 | D→A+C | ✅ 8/27 报告到；decision_policy v1.1 数值已标定（primary=RMSE_macro_norm，分类降 guardrail，hazard 禁作选优）；**待 C+导师 8/31 前签字冻结** | 签字项 |
| G-12 | DEC-001（M 轴）+ seeds v1.1 修订追认 + go/no-go 判定 | 导师+全组 | ✅ 8/27 决议定稿（18 文档 §四）：DEC-002 激活 / DEC-001 挂起 9/2 复议 / seeds 追认 / policy 冻结 / GO；站会仅作进度汇报 | — |
| G-15 | 冒烟闸门（CPU 假张量拦截坏编辑 + LLM 修复轮） | A | ✅ 8/27 晚：基线自检通过，历史 2 个失败编辑均可秒级拦截 | — |
| G-13 | 随机臂演练 2 trial，产物过 schema 校验 | A | ✅ 8/27 下午：rand-rep-000（R5）+ rand-obj-000（O2）两趟全链路 completed（axis_lock 正反例、smoke 五阶段、双腿评测、result schema、轴文件还原全过）；registry 占位路径不一致已修 | — |
| G-14 | C1 v1.0.2 升级验收（degenerate 语义按 V2、逐类 F1 输出、decision_policy_metrics 映射、合同元数据） | C→A | ✅ 8/27 16:20：golden 一致、基线 RMSE 不变（0.0481047）、policy 五键 1:1 对齐、degenerate 新语义解析无碍；SHA 已登记 | — |

**关键路径**：G-7（接线）与 G-11（seed 2027 跑）可并行；G-8 插在 G-7 中间做；全部走完即开正式 discovery，预计 **8/29–8/30 可开跑**（比 02 计划的 9/1 略有余量）。

## 待决事项现状

| 决策 | 状态 |
|---|---|
| DEC-001 M 轴 | ⬜ 8/27 站会；axis_lock 路径已备好（fusion/encoders） |
| DEC-002 tier-2 | 🔶 证据已到：seed42/43 双 seed 复现"forecast 预训练负迁移、classification 正迁移"——建议站会直接讨论激活 |
| DEC-003 ZBAD 预训练 | ✅ 已解决（ZBAD present: NO） |
| seeds v1.1（纳入 43） | 🔶 A 已修订留痕，待站会追认 |
| 主指标尺度 | ✅ contract v2 定案：归一化 [0,1] |
