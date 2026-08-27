# Closed-loop Auto Research Harness (CLH)

面向空管/航空气象的 **Closed-loop Auto Research** harness。协议严格复现 [Closed-loop Auto Research for Molecular Property Prediction](https://arxiv.org/pdf/2606.22731)（submitted-trial、file-level ablation lock、held-out certification）；数据面只走 [AeroWF Data Contract V1](../AeroWF/数据接口/AeroWF_v1_MODEL_TRAINING/DATA_CONTRACT_v1.md)。

> 智能体提出可证伪假设、只改一个研究轴、把流水线交给它不可控的 evaluator；搜索结束后冻结每轴 \(c_a\)，在从未读过标签的 held-out 上认证一次。主表必须 **validation | held-out 成对**，并区分 selection variance 与 distribution shift。

四轴合同与论文条款：[docs/README.md](docs/README.md)。试跑缺口与下一步：[docs/next-steps.md](docs/next-steps.md)。

## 架构

```
研究起点 (AeroWF DATA_CONTRACT / dummy 合成气象)
        │
        ▼
 ┌──────────────────────────────────────────────┐
 │  Closed-loop core（每 trial 一次）            │
 │  1 Hypothesis  →  2 Research Action           │
 │  3 Auto Experiment →  4 Independent Eval      │
 │  5 Lineage & Memory                           │
 │  Action space: Representation | Model         │
 │                | Physics | Data               │
 └──────────────────────────────────────────────┘
        │ 冻结每轴 c_a = argmax mean I_t^val
        ▼
 认证：Held-out temporal / spatial / event（只评一次）
        │
        ▼
 certification.json：成对 val/test + routed + 非迁移签名
```

Evaluator、划分、指标、泄漏过滤器在 agent 可编辑树之外。一次 trial 只能改当前轴允许的文件。认证分数不进入 `SessionLog.model_visible()`。

## 两个 profile

| 配置 | 域 | 数据 | 流水线 |
| --- | --- | --- | --- |
| `configs/dummy.toml` | 合成气象，协议单元测试 | 内存合成帧（ZBAA/ZSPD/ZGGG，holdout ZUUU） | `examples/dummy_research/pipeline` |
| `configs/atc.toml` | AeroWF | `AeroWF_v1_MODEL_TRAINING/release_v1`（ZBAA/ZSPD/ZSSS，空间 ZBAD） | `examples/aerowf_research/pipeline` |

ATC 默认 `max_samples_per_split = 128`；设为 `0` 使用完整 index。训练包若只有 `index.csv` 而无 `.npy`，harness 按冻结 min/max 物化合同形状张量。未挂官方 sealed 时，认证用 val 时间尾部（`inner_val_frac=0.20`, seed 42）做 temporal holdout，空间用契约气候的 ZBAD 探针；把评测包指到 `domain.sealed_root` 后改走 `sealed/temporal` 与 `sealed/spatial/ZBAD`。

## 运行

```powershell
Copy-Item .env.example .env
python -m pip install -e ".[dev]"
python -m pytest
python -m clh run --config configs/dummy.toml
python -m clh run --config configs/atc.toml --provider offline
python -m clh run --config configs/atc.toml --provider openai_compat
```

```env
CLH_API_KEY=sk-...
CLH_BASE_URL=https://api.deepseek.com/v1
CLH_MODEL=deepseek-chat
```

无 key 时走 `offline` specialist。DeepSeek / OpenAI / SiliconFlow / vLLM 等 Chat Completions 兼容网关均可。

## 论文条款 → 代码

| 论文条款 | 实现 |
| --- | --- |
| 轴 \(A=\{\mathrm{feature},\mathrm{model},\mathrm{data}\}\) | `representation` / `model` / `data`；ATC 另加 `physics` |
| 式 (1) \(c_a=\arg\max \overline{I}^{\mathrm{val}}\) | `research/certification.py` |
| 式 (3) \(I_t\)，等权聚合 | `research/reward.py` |
| Routed，\(\tau=0.005\) | `certification.json` 的 `routed` |
| File-level ablation lock | `research/axis_lock.py` |
| 泄漏过滤 | AeroWF：`domain/aerowf/leakage.py`；dummy：`domain/atc/leakage.py` |
| `write_external_data` | `plugins/compose.py`（仅 catalog 源） |
| 非迁移签名 | `classify_signature` |

端点 \(\mathcal{T}\)：`overall.mae`、各机场 `.mae`、`hazard.csi`。搜索奖励只来自可见 val。

## 数据边界（MODEL_HANDOFF_v1）

**允许（搜索 / 训练）**：`pretrain/train` · `pretrain/val` · `trainval/train` · `trainval/val`

**禁止**：`pretrain/test` · `sealed/**` · ZBAD

张量：`runway.npy` `(N,4,96,11)` 通道顺序冻结；有效跑道只看 `runway_mask.npy`；归一化只读 `train_stats_v1.json` / `pretrain_stats_v1.json`。PRE2020 的 `weather_label.npy` 不得作为 SSL 目标。

## 目录

```
src/clh/
  research/     loop, axis lock, evaluator, certification, specialists
  domain/aerowf DATA_CONTRACT loader · leakage · path guards
  domain/dummy  合成气象
examples/aerowf_research/pipeline/
examples/dummy_research/pipeline/
configs/        dummy.toml, atc.toml
docs/axes/      协议与四轴合同
```
