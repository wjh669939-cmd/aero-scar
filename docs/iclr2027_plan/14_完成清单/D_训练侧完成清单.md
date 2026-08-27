# D（AeroWF 训练执行侧）完成清单

> 状态基准：2026-08-26 深夜。v2 全流程已在冻结数据上双 seed 跑通并全审计通过。

## 已完成 ✅

| 项 | 内容 | 对照 |
|---|---|---|
| 官方代码修复 | 语法/遮挡概率/MSE 分母修复；SoftDTW CUDA 验证可运行 | 08 前置 |
| 冻结数据接入 | release_v1 全量加载；边界断言（只碰 pretrain train/val + trainval train/val） | 08 红线 1 |
| D2 任务头 | Forecast（contract v2：15/60/120min × wind_x/y，时间戳对齐）+ Classification（21→3、AllowedContextEncoder 防泄漏、weighted CE + ignore_index） | 08 D2 |
| 全流程控制器 | `aerowf_full_pipeline_v2.py` 五阶段一键；六条件成功判定自审计（status/test_used/checkpoint SHA/keys） | 08 D5 前置 |
| seed 42 全流程 | 预训练 100ep + 四下游，分阶段跑通（分类 Pretrained F1=0.809 / 预测 Scratch RMSE=0.0485） | 08 D3/D4 |
| seed 43 全流程 | 一键 pipeline（分类 Pretrained F1=0.735 / 预测 Scratch RMSE=0.0481），handoff 包齐 | 08 D3/D4 |
| **D4 核心问题初答** | 双 seed 一致：**forecast 预训练负迁移、classification 正迁移**——KDD T+1 异常在修复版代码 + 冻结数据上复现（任务级）。DEC-002 证据已到 | 08 D4 ★ |
| 目录树 + CODE_MAP | axis_lock 四个文件级问题全部回答（8/27 交付提前完成） | 08 v1.1 |
| 作废处理 | scratch rand25 30ep 指标作废（冻结前数据+评测机场混入），已入 ledger | 08 v1.1 |
| HANDOFF 手册 | 环境快照（3090/torch2.5.1+cu124）、调用模板、成功/失败判定、输出保护规则 | 新增交付 |

## 剩余

| 项 | 内容 | 依赖 | 时点 |
|---|---|---|---|
| 🔶 D1 方差报告 | **只差 seed 2027 一跑（~2.7h）**：42✅/43✅/2027⬜ 三 seed 出"效应量 vs 噪声"报告（配对口径）；seeds v1.1 修订随站会追认 | seeds v1.1 | 8/27 |
| ⬜ 抽薄改造执行 | trial_features.py / trial_objective.py 拆分（A 出方案）；抽薄后 seed43 复跑一致性验证 | A 方案 | G-8，0.5 天 |
| ⬜ G1 接线联调 | 配合 A：CLH 子进程调用 pipeline 的 smoke（--pretrain-epochs 1 模板已备） | G-8 | G-7 |
| ⬜ D3 收尾 | parent 5 seed（差 3407/5519）+ Persistence/逻辑回归/多数类参照入 C1 evaluator 同一口径 | D1 后 | 8/31 前 |
| ⬜ D5 discovery 值守 | 排队/失败重跑/资源记账（GPU 与 discovery 共享 3090 的排期要在站会定） | discovery 开跑 | 9/1– |
| ⬜ D6 候选确认 | 各臂 top 候选 5 seed 配对确认 | discovery | 9/9–9/10 |

## 待排查（不阻塞，但进论文前必须有结论）

- **分类 seed 敏感性**：Pretrained Macro-F1 增益 +7.7%（seed42）vs +0.7%（seed43），HAZARD F1 0.789 vs 0.527——在 support=94（仅 ZBAA）的类上这可能就是小样本方差。seed 2027 跑完即有三点估计；**决策含义：分类（尤其 hazard 类）不适合做主 endpoint，建议只做 guardrail**，主 endpoint 用预测 RMSE_macro_norm（三 seed 内差 <0.001，极稳）；
- 预训练耗时差异（seed42 53min vs seed43 79min）：确认是否 dataloader/机器负载差异，记入资源账本。

## 红线提醒

- 任何训练路径禁触 sealed/ZBAD/pretrain/test（手册 §9 已自设，保持）；
- 调参只在冻结校准预算内（≤10 次、tuning_log.jsonl 留痕）；
- 输出目录一 trial 一目录，不复用不覆盖。
