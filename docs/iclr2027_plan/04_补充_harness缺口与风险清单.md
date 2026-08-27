# 补充文档：Harness 缺口清单与工程任务分解 v1.0

> 日期：2026-08-23
> 性质：**补充文档**（由 AI 协作者建议增补，非导师要求的三份主文档之一）
> 用途：(1) 让导师了解"闭环愿景图"与现实实现之间的确切距离；(2) A（framework 负责人）的工程任务清单与排期依据

---

## 一、闭环愿景图逐组件对照

对照图中"Closed-loop Auto Research 核心闭环 + Certification"各组件，现状分三级：

| 图中组件 | 现状 | 证据 / 缺口 |
|---|---|---|
| ① Hypothesis 假设生成 | 半成品 | proposal 机制在，Radar 上只跑过单变量 alpha 提案；面向 AeroWF 的证据注入提示（failure slices → 假设）**未实现** |
| ② Research Action 动作选择 | 半成品 | Action schema（含 hypothesis / falsification / 预算 / rollback 引用）有协议定义；双轴 config/code-diff 执行器**未实现** |
| ③ Auto Experiment 自动实验 | 骨架已验证 | 真实 GPU executor、preflight digest gate、失败重试记账均已实战；**AeroWF 训练入口未接入** |
| ④ Independent Evaluation 独立评估 | 模式已验证 | 冻结 evaluator + subprocess 隔离 + invalid/failed 三态在 Radar v1.3 上实战过；**AeroWF endpoint 网格 evaluator 未实现** |
| ⑤ Lineage & Memory | 已验证 | research trace、artifact manifest、SHA-256、H3 单次 finalization 均验收 |
| DecisionPolicy + 回滚 | 已实战 | Radar 上 alpha=1.5 因 guardrail 越界被自动拒绝回滚 |
| 多轮无人值守迭代 | **从未实战** | H1（预算持久化）/H2（阶段重入）已验收但只在测试中用过；正式 campaign 历史上没有跑过第 3 轮 |
| accepted → promote → freeze | 仅合成测试 | H5 通过，真实任务中从未出现 accepted 候选 |
| Held-out Test 一次性认证 | 机制已建 | test lock + Custodian + 访问日志模式在 Radar/SEVIR 合同中定义；**三类环境（temporal/ZBAD/event）实例化未做** |
| 图中 Data Axis / Model Axis / Physics Axis | **本篇不做** | 砍轴决策见 01 文档 §3.4 |

**结论：闭环的"控制面"成立且经过实战 + H1–H5 加固；"领域面"（AeroWF 的数据/训练/评测/动作四件套）为零，需新建。**

## 二、工程任务分解（A 的排期，共约 6–8 人天）

| # | 任务 | 内容 | 估时 | 截止 |
|---|---|---|---|---|
| T1 | harness 迁移部署 | SimpleAutoResearch + 依赖装到 3090 机器；680 项测试重跑，记录基线 | 0.5 天 | 8/24 |
| T2 | AeroWF 数据合同 | `[B,R,T,F]` + runway_mask 的 dataset access 层；train/val manifest + digest；test/ZBAD 路径进禁访 token 列表 | 1 天 | 8/26 |
| T3 | AeroWF 训练入口 | 复用官方（已修 bug）训练循环，包成 harness executor 可调的入口：spec 进、checkpoint+predictions 出、资源用量记账 | 1 天 | 8/27 |
| T4 | 预测合同 + evaluator 接线 | predictions 格式（按任务分别定义）+ sample_id 对齐；C 的 endpoint 网格 evaluator 以 subprocess 方式接入 | 1 天 | 8/29 |
| T5 | 双轴动作空间 | 白名单路径 axis lock（复用现有 diff 检查）；Representation/Objective 各 4–6 个预实现动作模板 + 自由 config-diff 动作 | 1.5 天 | 8/30 |
| T6 | LLM proposal 管线 | 证据包（基线 failure slices + lineage 摘要）→ 结构化提案（hypothesis/falsification/预算）；重复动作检测复用 memory | 1 天 | 8/31 |
| T7 | 端到端 smoke ×2 | 真实数据完整闭环 2 个 trial + 1 个越权动作被拒的反例 | 0.5 天 | 8/31 |
| T8 | 随机对照采样器（E 主做，A 审） | 同一动作空间的合法随机采样，绕过 LLM 直接进 executor | 0.5 天 | 9/1 |

依赖关系：T2 依赖 B 的重切分（8/25）；T4 依赖 C 的 evaluator v1.0（8/29 冻结前可用草案版联调）。

## 三、风险登记册（按暴露度排序）

| # | 风险 | 概率 | 影响 | 缓解 | 触发的分支 |
|---|---|---|---|---|---|
| R1 | 3-seed 方差 ≥ 可发现增益 | 中高 | 主叙事死亡 | 8/26 gate 前置；配对同 seed 比较可部分抵消 | 分支 B |
| R2 | issue-time 泄漏存在于原始管线 | 中 | 基线重做、数字全变 | 第一周专项审计；发现即修，如实写入论文 | 进度顺延，冻结日不动 |
| R3 | 多轮无人值守闭环在真实任务上暴露未测路径 | 中 | discovery 中断 | 阶段 2 前 3 天人工值守；H1/H2 恢复机制已验收 | 分支 C |
| R4 | LLM 提案质量低（重复/不可执行/纯调参） | 中 | RQ1 答案为否 | 预实现动作模板保底；重复检测；这本身是可报告结果 | 分支 D |
| R5 | 预训练复现太贵或数据缺失 | 中 | 失去最佳故事线 | tier-2 从一开始就是 P1；Scratch 主线不受影响 | 砍 tier-2 |
| R6 | ZBAD 认证结果很差 | 中 | 空间迁移主张受限 | 这是发现不是失败：non-transfer 归因 + few-shot 语境（KDD 论文中 ZBAD 本就是难迁移目标） | 分支 E |
| R7 | 人员单点（A 是 harness 唯一熟手） | 中 | 关键路径阻塞 | T1–T7 全部产物落文档；D 作为 harness 二线从阶段 2 起跟班 | — |
| R8 | 3090 机器故障/占用冲突 | 低 | 全线停摆 | 每晚同步 checkpoint 与 artifact 到本机（radar 服务器）；预算含 30% 冗余 | — |

## 四、给导师的一句话总结

> 闭环不是要"从头搭"，控制面（提案-执行-评测-决策-谱系-回滚-测试锁）已在雷达任务上实战并完成五项加固；本篇要补的是 AeroWF 领域面四件套（数据合同、训练入口、网格 evaluator、双轴动作），约一周工程量，8/31 以两个端到端 trial 验收。真正的不确定性不在工程，在 8/26 的 seed 方差判定——它决定我们写"发现"还是写"测量"。
