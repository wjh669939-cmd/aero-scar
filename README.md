# AeroWF-SCAR: Shift-Certified Auto Research for Aerodrome Weather Forecasting

面向 ICLR 2027 的闭环自动科研（Closed-loop Auto Research）项目团队归档。
以 KDD 2026 AeroWF 为冻结基线，在文件级轴锁、冻结评测器与一次性隐藏认证的约束下，
由 LLM 智能体提出可证伪假设、修改受控代码面、运行真实训练并接受独立评测。

> 本仓库为**团队内部归档**：不含任何训练/评测数据、认证数据、模型权重、私有配置或密钥。
> 数据交付与认证封存由数据侧（B）与评测侧（C）线下保管，以 SHA-256 清单对账。

## 目录结构

| 目录 | 内容 | 归属 |
|---|---|---|
| `harness/` | CLH 闭环框架（proposal → axis lock → subprocess 执行 → 独立评测 → lineage → 认证），含全部单测 | A |
| `contract/` | 冻结契约区：trial/result schema、axis_lock、action registry、decision_policy、seeds、下游任务协议 v2、test_lock_state（脱敏版） | 全组 |
| `tools/` | 契约工具链：schema 校验、axis_lock 引擎、随机臂采样器、提案解析器、上下文组装器（隐藏 token 拦截）、pipeline 执行器（六条件判定）、预测适配器、evaluator 客户端 + 55 项单测 | A |
| `discovery/` | 正式 discovery 驱动（两段式 LLM + CPU 冒烟闸门）、G-13 演练脚本、trial 记录与 lineage、parent 参考指标（仅 metrics，不含预测文件） | A |
| `model_side/` | AeroWF 基线模型代码（models/utils/训练脚本）与下游 v2 流水线（G-8 抽薄后：`trial_features.py` / `trial_objective.py` 为唯二可写轴面） | D |
| `evaluator/` | C1 冻结评测器当前版（v1.0.2，含 golden 与边界测试报告）、历史版归档（去 golden 数据）、C2 认证封存哈希、危险事件切片定义 | C |
| `docs/` | 规划文档（任务书、完成清单、联调结论、站会决议）与各侧交接手册/审计报告 | 全组 |

## 不在仓库中的东西（有意排除）

- 冻结数据交付包（`AeroWF_v1_MODEL_TRAINING` / `EVALUATION`）与一切 npy/npz 训练数据；
- 模型权重（`*.pth`）与训练产物目录（`results/`）；
- C 私有配置（val 真值、私有 mask、认证数据、锁与访问日志）；
- 认证封存内容本体（仅保留 SHA-256 对账清单）；
- API key 与机器凭据（运行时经环境变量/命令行注入，不落盘）。

## 复现入口

- 单测：`cd tools && python -m pytest tests/`（55 项）与 `cd harness && python -m pytest tests/`；
- 冻结基线全流程：`model_side/aerowf_downstream_v2/src/aerowf_full_pipeline_v2.py`（六条件成功判定见 `docs/model_side/HANDOFF.md` §6）；
- discovery：`discovery/discovery_runner.py --plan representation,objective_tier1 --seed 42`（需 DeepSeek key 与 C 评测器私有配置就位）。

## 关键治理事实（截至 2026-08-27）

- 评测器 v1.0.2 冻结（计分位置值域 + 预注册容忍带 + decision_policy_metrics 直接映射）；
- decision_policy v1.1 阈值全部由三 seed（42/43/2027）方差报告标定，先于任何候选结果冻结；
- 激活轴：R + O-tier1 + O-tier2；M 轴 conditional（DEC-001，9/2 复议）；
- parent seed42 四腿参考已过冻结评测器（forecast scratch RMSE_macro_norm 0.048471）；
- test 三认证环境（temporal/spatial/event）LOCKED，一次性解锁窗口 2026-09-11~12。
