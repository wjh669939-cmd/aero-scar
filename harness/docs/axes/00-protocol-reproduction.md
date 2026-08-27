# Closed-loop Auto Research 协议复现（对标 arXiv:2606.22731）

> 论文：[Closed-loop Auto Research for Molecular Property Prediction](https://arxiv.org/pdf/2606.22731)  
> 领域适配：AeroWF / 航空气象 · ATC  
> 数据契约：`任务测试/AeroWF/数据接口/`

本文档把论文实验设计 **逐条落到 AeroWF harness**，不稀释为“自动调参”。四轴细则见同目录：

| 论文轴 \(A\) | AeroWF 轴 | 说明文档 |
| --- | --- | --- |
| feature | representation | [01-axis-representation.md](01-axis-representation.md) |
| model | model | [02-axis-model.md](02-axis-model.md) |
| （领域扩展） | physics / domain | [03-axis-physics.md](03-axis-physics.md) |
| data / external evidence | data | [04-axis-data.md](04-axis-data.md) |

论文原文只有 **feature / model / data** 三轴。Physics 轴是 ATC 任务图与导师方案中的第四轴：在 **不改表示、不改估计器结构、不改外部证据** 的前提下，只改物理约束与目标函数。归因规则与三轴相同（file-level ablation lock）。

---

## 1. 核心主张（必须保留）

Closed-loop Auto Research 的基本单元是一条 **谱系（lineage）**：提出假设 → 可执行干预 → 提交给 **agent 不可控** 的 evaluator → 用实测结果更新后续假设。

与 AutoML 的差别：

- AutoML：固定数据集 + 预定义搜索空间内的超参/算法选择。
- Auto Research：可以改 **表示 / 模型代码 / 外部证据 /（本域）物理目标**，并问：**验证集上选中的改进，是否在从未读过标签的 held-out 上仍成立？**

可靠性问题用 **certify-after-search** 解决，而不是在搜索中限制 holdout 访问（constrain-during-search）。

---

## 2. 严格复现条款（论文 §3）

### 2.1 单轴归因（§3.1）

记端点集合 \(\mathcal{T}\)，轴集合 \(A\)。对轴 \(a\)，在可见验证集上选：

\[
c_a = \arg\max_c \frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}} I_t^{\mathrm{val}}(c)
\]

每个轴只报告 **一个** best-aggregate 配置 \(c_a\)。该配置的验证回报与 held-out 回报成对：

\[
R_{t,a}^{\mathrm{val}} = I_t^{\mathrm{val}}(c_a),\quad
R_{t,a}^{\mathrm{test}} = I_t^{\mathrm{test}}(c_a)
\]

**Routed**：对每个端点，在验证回报 \(\ge \tau=0.005\) 的轴中取验证最优轴，再读该轴的 held-out 回报。选择 **永不使用** test 标签。

### 2.2 归一化提升（§3.2 式 3）

\[
I_t =
\begin{cases}
(s_t - b_t)/|b_t| & \text{越高越好} \\
(b_t - s_t)/|b_t| & \text{越低越好}
\end{cases}
\]

aggregate trial score = 端点 \(I_t\) 的等权平均。baseline 在每个 suite 上单独校准，使 aggregate \(I=0\)。

### 2.3 Held-out 认证协议（§3.3）

搜索结束后：

1. **冻结** \(c_a\)（特征代码、模型代码、已准入外部数据、物理/目标代码）。
2. 在内部 train 上 **从头重训**。
3. 对 held-out test **只评一次**。
4. 内部 train/val 划分与搜索期奖励所用划分一致（论文：scaffold 80:20，seed 42）。
5. data 轴合并必须 **重放** 同一泄漏过滤器（可读 test 结构/时间键，不可读 test 标签）。
6. harness **不得修改** 已选配置。

AeroWF 对应的隐藏分区（见数据接口）：

| 认证类型 | 论文对应 | AeroWF 路径 / 边界 |
| --- | --- | --- |
| 搜索可见 val | internal validation | `release_v1/trainval/val`、`pretrain/val` |
| temporal held-out | outer test | `release_v1/sealed/temporal`（训练包外） |
| spatial held-out | 分布外端点 | `release_v1/sealed/spatial/ZBAD` |
| event / 极端 | （气象扩展） | 事件分组留出；不得在搜索中读标签 |
| PRE2020 SSL test | 预训练 held-out | `release_v1/pretrain/test`（训练侧禁止） |

### 2.4 Submitted-trial 循环（§3.4）

一次 trial 四步（任务图扩展为五步含 memory）：

1. 读 lineage → 提出假设  
2. 做 **单轴** 可执行修改  
3. 提交流水线  
4. harness 评全部可见端点、赋 status、写入 lineage  

约束：

- 同 backend 的 specialists，按可编辑表面分组。
- **File-level ablation lock**：每次 subprocess 前，从 pristine 恢复非目标轴文件。
- evaluator / 指标 / 泄漏过滤器 **在可编辑树之外**。
- 轴内 trial **串行**，后一 trial 必须读到更新后的 lineage。
- 各轴 **等预算**（论文 TDC≈100/轴；迁移研究≈30/轴）。首版禁止 online bandit 重分配预算（否则轴能力与预算策略纠缠）。

### 2.5 强基线（§3.6）

论文：未编辑流水线 = MapLight 风格固定表示 + CatBoost，无逐任务调参。

AeroWF：未编辑流水线 = 冻结 AeroWF 基线 + `DATA_CONTRACT_v1` 张量契约 + `train_stats_v1.json` / `pretrain_stats_v1.json` 冻结归一化。**禁止**用 val/test/ZBAD/sealed 重 fit 统计量。

### 2.6 泄漏安全外部证据（§3.7）

论文三层过滤器 → AeroWF 气象版映射见 [04-axis-data.md](04-axis-data.md)。agent **不可绕过** evaluator-owned filter。

### 2.7 非迁移签名（§4）

成对报告时必须区分：

| 签名 | 验证 | held-out | 论文例 |
| --- | --- | --- | --- |
| selection variance | 大增益 | ≈0 | TDC model：val 0.041 → test 0.003 |
| distribution shift | 正增益 | 负增益 | Polaris data：val +0.022 → test −0.019 |

主表必须 **validation | held-out 并排**，不得只报验证最优。

---

## 3. AeroWF 搜索可见 vs 密封边界（硬规则）

摘自 `MODEL_HANDOFF_v1.md` / `RELEASE_NOTES_v1.md`：

**允许（搜索 / 训练）**

- `release_v1/pretrain/train` · `pretrain/val`
- `release_v1/trainval/train` · `trainval/val`

**禁止（agent 与训练团队）**

- `release_v1/pretrain/test`
- `release_v1/sealed/**`
- ZBAD 下游评测数据

密封数据不得用于：梯度优化、early stopping、超参、特征选择、阈值、归一化拟合、结构选择、checkpoint 选择、或根据 test 分数人工迭代。

---

## 4. 当前实现状态（2026-08-25）

配置：`configs/dummy.toml`（合成域）与 `configs/atc.toml`（AeroWF）。ATC 流水线：`examples/aerowf_research/pipeline/`。

**端点 \(\mathcal{T}\)**（`domain/metrics.py`）：`overall.mae`（↓）、各训练机场 `{ICAO}.mae`（↓）、`hazard.csi`（↑）。trial 分数 = 这些 \(I_t\) 的等权平均。

**内部划分**：搜索奖励只用可见 val。论文 scaffold 80:20 / seed 42 对应 ATC 的 `inner_split_seed = 42`。官方 sealed 未挂载时，从 `trainval/val` 按时间切出最后 `inner_val_frac=0.20` 作为 evaluator-owned temporal holdout（该尾部 **不** 进入搜索 val）；空间为 ZBAD 契约气候探针。挂上 `domain.sealed_root` 后改用 `sealed/temporal` 与 `sealed/spatial/ZBAD`。

**基线**：dummy = persistence（把第一列当预报）；AeroWF = 对 mask 池化后的 **上一时刻** 风速做 persistence。二者都是「强紧凑、未逐任务调参」的起点，不是完整神经网络。

**产物**：`runs/<id>/summary.json`、`certification.json`（含 `axes` / `routed` / `signature`）、`session.jsonl`。认证事件对 specialist 不可见。

---

## 5. Harness 落地对照

| 论文机制 | CLH 实现 |
| --- | --- |
| lineage | `SessionLog`（model-visible ⇔ logged；认证分数对 agent 隐藏） |
| ablation lock | `research/axis_lock.py` |
| independent evaluator | `research/evaluator.py` |
| \(I_t\) / \(c_a\) / routed | `research/reward.py` + `research/certification.py` |
| held-out certification | `research/certification.py` |
| AeroWF 数据契约 | `domain/aerowf/`（`DATA_CONTRACT_v1` + `MODEL_HANDOFF_v1`） |
| leakage filter | `domain/aerowf/leakage.py`（气象版）；dummy：`domain/atc/leakage.py` |
| `write_external_data` | `plugins/compose.py` |
| specialists | `research/specialist.py` |
| 等预算串行搜索 | `research/loop.py` + `configs/*.toml` |

---

## 6. 汇报单位（禁止作弊）

- 主文单位 = 每轴一个 \(c_a\) 的 **成对** val/test，不是“在所有 trial 上对 test 取 max”。
- 端点级 trial-max 只能进附录探索，**不得**进入配置选择。
- data 轴：过滤器通过 ≠ 分布匹配；Polaris 式负迁移必须可报告。
