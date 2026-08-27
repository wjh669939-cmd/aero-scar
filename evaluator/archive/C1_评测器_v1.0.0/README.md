# C1｜评测器 v1.0

供 CLH 以子进程调用：

```bash
python -m aerowf_evaluator --predictions predictions.npz --trial-meta trial_meta.json --split val --out-dir out
```

调用方只能提供预测、trial 元信息、`val` 和输出目录；数据路径只在 C 的私有配置中解析。评测器不 import CLH，CLH 也不 import 本包，双方只通过文件和子进程交互。
`trial_meta.json` 必填：`trial_id`、`arm`、`seed`、`task`、`checkpoint_digest`。

正式任务：风预测（T+1/T+4/T+8 × wind_x/wind_y × MAE/RMSE）、三分类（Macro-F1、三类 CSI、CSI_macro）和固定 mask 插补（四场景 MSE/MAE）。插补 `pred` 形状为 `[sample, runway_slot, scenario, minute, channel]`，其中 scenario 的固定顺序为 `random_25`、`random_50`、`random_75`、`feature`；一次运行输出 24 个插补 endpoint。不含 Crosswind、C2 认证数据或 C3 接受阈值。

每次运行输出 `metrics.json`（逐机场 endpoint、各类 overall、三态和异常计数）与 `evaluation_manifest.json`（代码、配置、数据口径和输入摘要）。`invalid` 表示提交接口错误；`failed` 表示评测器或 C 私有环境故障；两者均为 `not_evaluated`。

`golden/` 是 CLH 适配器的固定预测向量和对应的完整期望 `metrics.json`。由于正式接口要求完整 val 的 `sample_id`，golden 保留完整样本编号；其中不含认证数据。
