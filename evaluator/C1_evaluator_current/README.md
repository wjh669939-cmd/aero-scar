# C1｜评测器 v1.0.2

供 CLH 以子进程调用：

```bash
python -m aerowf_evaluator --predictions predictions.npz --trial-meta trial_meta.json --split val --out-dir out
```

调用方只能提供预测、trial 元信息、`val` 和输出目录；数据路径只在 C 的私有配置中解析。评测器不 import CLH，CLH 也不 import 本包，双方只通过文件和子进程交互。
`trial_meta.json` 必填：`trial_id`、`arm`、`seed`、`task`、`checkpoint_digest`。

正式任务：风预测（T+1/T+4/T+8 × wind_x/wind_y × MAE/RMSE）、三分类（Macro-F1、三类 CSI、CSI_macro）和固定 mask 插补（四场景 MSE/MAE）。插补 `pred` 形状为 `[sample, runway_slot, scenario, minute, channel]`，其中 scenario 的固定顺序为 `random_25`、`random_50`、`random_75`、`feature`；一次运行输出 24 个插补 endpoint。不含 Crosswind、C2 认证数据或 C3 接受阈值。

每次运行输出 `metrics.json`（逐机场 endpoint、各类 overall、三态和异常计数）与 `evaluation_manifest.json`（代码、配置、数据口径和输入摘要）。`invalid` 表示提交接口错误；`failed` 表示评测器或 C 私有环境故障；两者均为 `not_evaluated`。

v1.0.1 修订：连续预测的值域检查只作用于真实且实际计分的位置；虚拟跑道槽位（padding）和其他不计分位置跳过。计分位置的常规范围仍是 `[0,1]`：若全部越界值都在 `[-0.1,1.1]` 内，且越界数量不超过该任务全部计分位置的 `0.01%`，则保留原始预测值并正常计分，同时在 `anomaly_counts.out_of_range` 如实记录；其余越界仍为 `invalid`。评测器绝不 clip、截断或修正预测值。

变更理由：该修订把检查范围与实际指标计算范围对齐，并预先固定线性回归头可能出现的极少量轻微溢出处理。它不改变预测值、模型、指标定义或候选结果；所有超出容忍带或数量阈值的提交仍会被拒绝。

v1.0.2 一次修订：为与 V2 合同和已冻结的 DecisionPolicy 指标名称一致，`support>0` 的 GOOD、PRECIP、CLASS_HAZARD 均保留逐类 F1 和 CSI 数值；仅 `support=0` 时两者为 `null`。事件率为 0 或 1 的 endpoint 仍标记 `degenerate=true`，但只从 overall 排除，不抹去数值。`metrics.json` 输出逐类 F1（包括 `f1_class_hazard`）及相应 overall，并新增 `decision_policy_metrics`：`RMSE_macro_norm`、`MAE_macro_norm`、`classification_macro_f1`、`classification_csi_macro`、`hazard_class_f1` 与草案同名；每项均带对应 C1 overall 字段的 `source_metric`、数值、CI 与退化标记。`metrics.json`、`evaluation_manifest.json` 均记录 `data_release`、`metric_scale`、`forecast_protocol`。本次修订不基于候选结果，不修改 DecisionPolicy 草案的任何规则或数值；当前 v1.0.2 是该版本唯一正式冻结副本，v1.0.1 已保留在归档目录。

`golden/` 是 CLH 适配器的固定预测向量和对应的完整期望 `metrics.json`。由于正式接口要求完整 val 的 `sample_id`，golden 保留完整样本编号；其中不含认证数据。
