# 轴 4 · Data / External Evidence（对应论文 Data Axis）

> 论文轴名：`data`（external evidence）  
> AeroWF 轴名：`data`  
> **数据侧真源**：`任务测试/AeroWF/数据接口/`  
> 协议总览：[00-protocol-reproduction.md](00-protocol-reproduction.md)

---

## 1. 论文定义（严格）

Data 轴在 **data_only** 模式下，经受控接口增加外部有标签证据；ablation lock **禁止**同时改 feature / model。

论文三层泄漏过滤器（§3.7，agent 不可绕过）：

1. **标准化身份去重**：脱盐、中和、InChIKey；去掉与 train/val/test 重叠行。  
2. **同源整文件拒绝**：外部文件与 endpoint test 的骨架重叠 **> 5%** → 整文件拒绝。  
3. **近邻过滤**：ECFP4 Tanimoto \(\ge 0.9\) 相对 test 的分子剔除。

每笔 merge 记录 source、overlap、分层计数。过滤器通过是迁移的 **必要不充分** 条件（Polaris data：val +0.022 → test −0.019）。

工具面：`write_external_data` 一类受控写入（附录 S5）；agent 不能直接任意读盘合并。

---

## 2. AeroWF 数据接口（本轴唯一允许的数据面）

### 2.1 目录与角色

根路径：`任务测试/AeroWF/数据接口/AeroWF_v1_MODEL_TRAINING/`

| 路径 | 角色 | Data 轴是否可“使用” |
| --- | --- | --- |
| `release_v1/pretrain/train` | SSL 优化 | 是（搜索可见） |
| `release_v1/pretrain/val` | SSL 验证 / 选型 | 是（不可当 test） |
| `release_v1/trainval/train` | 下游优化 | 是 |
| `release_v1/trainval/val` | 下游 val / 模型选择 | 是（搜索奖励） |
| `release_v1/pretrain/test` | SSL held-out | **否**（训练包外 / 禁止） |
| `release_v1/sealed/temporal` | 时间外推认证 | **否**（仅认证阶段，evaluator 持有） |
| `release_v1/sealed/spatial/ZBAD` | 空间外推认证 | **否** |
| ZBAD 下游其它评测 | 密封 | **否** |

机场（训练可见）：**ZBAA / ZSPD / ZSSS**。空间认证机场：**ZBAD**（不得进入搜索期数据合并）。

规模锚点（`RELEASE_NOTES_v1.md`）：

- Downstream train+val：68,481（train 56,598 / val 11,883）  
- Sealed temporal：12,198；sealed spatial ZBAD：27,101  
- PRE2020 三机场 train/val/test：97,914 / 20,982 / 20,982  

### 2.2 张量与归一化契约（只读）

见 `DATA_CONTRACT_v1.md`：

- 跑道窗：`N_max=4`, `time_steps=96`, `channels=11`  
- 外生连续 3 维 + 类别/二元 4 项  
- 下游 / PRE2020 分别使用 `train_stats_v1.json`、`pretrain_stats_v1.json`  
- **禁止**用 val、test、ZBAD、sealed、downstream 2025 去 fit PRE2020 统计量  

### 2.3 文件级锁

| 允许修改 | 禁止修改 |
| --- | --- |
| `data.py`, `external_manifest.json`（仅声明源 ID / 清单，不直写 sealed） | `features.py`, `model.py`, `physics.py`, `objective.py`, 冻结 `.npy` 内容与 stats JSON |

Agent 提交的是 **manifest / 源声明**；真正的过滤、合并、去重由 **evaluator-owned** 过滤器执行。

---

## 3. 气象版泄漏过滤器（论文 §3.7 → AeroWF）

不照搬 InChIKey / Tanimoto；改用时空与来源语义（导师方案 + 数据契约）：

| 层 | 规则 | 对应论文层 |
| --- | --- | --- |
| L1 精确身份 | `(airport, sample_id)` 去重；与 train/val/test 键冲突则删行 | identity de-dup |
| L2 同源拒绝 | 外部源与 **test** 样本键重叠率 \(>5\%\) → **整源拒绝** | same-source >5% |
| L3 近重复 | 同机场且 \(\lvert\Delta t\rvert \le 111\) min（96 min 窗 + 15 min 采样）相对 hidden 样本 → 剔行 | near-analogue |
| L4 因果 | 外部时间戳 \(\le\) 训练期最大时间 | （气象必需） |
| 审计 | `LeakageDecision`：source、overlap、removed/kept、layers；写入 evaluator 日志 | merge log |

实现：`src/clh/domain/aerowf/leakage.py`。过滤器可读 hidden 的 **键与时间**（用于去污），**不可**把标签或分布统计返回给 agent。

**Catalog 源**（`write_external_data` / `extra_source_ids()`，禁止任意路径）：

| source_id | 含义 | 预期审计 |
| --- | --- | --- |
| `pretrain_train` / `matched_climate` | `release_v1/pretrain/train`（搜索可见；PRE2020 标签不作 SSL 目标） | 通常可准入（与 2025 trainval 键不重叠） |
| `shifted_climate` | 错配气候探针（ZJHK） | 可准入；spatial 上可能 distribution shift |
| `leak_val` / `same_source_leak` | 与 val+temporal 重叠的同源陷阱 | **必须** L2 拒绝 |

Agent 只改 `data.py` 与 `external_manifest.json`。真正 merge 在 evaluator 内重放同一过滤器。

---

## 4. Trial 合同

Data trial 假设须说明：

- 引入何种外部 / 增广证据（更多机场·年份、雷达、闪电、NWP、传感器失效模拟、危险事件重采样、气候区预训练等）；  
- 预期改善的 failure slice；  
- 若被 L2 拒绝应如何改方向（不可改过滤器）。

执行：

1. 写 `external_manifest.json` + `data.py` 中的受控 `extra_source_ids()`（或等价接口）。  
2. Evaluator 跑 L1–L5；拒绝则 trial=`rejected`，仍记 lineage。  
3. 仅在可见 train 上合并准入行，在 `trainval/val` 打分。  
4. 冻结 \(c_{\mathrm{data}}\) 时 **重放** 同一过滤器后再认证。

SSL 红线：PRE2020 的 `weather_label.npy` 仅为 provenance，**不得**当 SSL 目标。

---

## 5. 评分与认证

- 选择：\(c_{\mathrm{data}}=\arg\max_c \overline{I}^{\mathrm{val}}(c)\)（仅统计准入且跑通的 trial）。  
- 认证分区：优先 `sealed/temporal`、`sealed/spatial/ZBAD`、事件留出。训练包未含 sealed 时，见 [00-protocol-reproduction.md](00-protocol-reproduction.md) §4 的 fallback（不得把 fallback 写成官方 sealed 分数）。  
- 必须报告：准入/拒绝审计（对齐论文 Table S7）。  
- 若 val 升、spatial/temporal 降：标记 **distribution shift**，不得称为可泛化发现。  
- 加载器：`domain/aerowf/world.py`；有 `runway.npy` 则读张量，否则按 `index.csv` + 冻结 stats 物化。路径含 `sealed` / `ZBAD` / `pretrain/test` 的搜索加载直接 `EvaluatorError`。

---

## 6. 首批可检验假设（示例）

1. **气候匹配增广**：引入气候相近机场的额外小时 → 预期 source-airport MAE↓；若实际为热带/高原错配源，预期 spatial 认证反转。  
2. **危险事件重采样**：不改模型，只改采样分布 → CSI↑；负对照 = 均匀采样。  
3. **传感器失效模拟**：训练期掩码增强 → 缺失鲁棒性↑。  
4. **同源陷阱**：故意声明与 sealed 高重叠的再发布源 → 必须被 L2 整源拒绝（过滤器回归测试）。

---

## 7. 禁止事项

- Agent 直接打开 `sealed/**` 或 ZBAD 评测包做训练。  
- 绕过 manifest，手工拷贝外部文件进 train 目录。  
- 用 test 标签做数据筛选后声称“data 轴发现”。  
- 修改冻结归一化 JSON 或通道契约冒充“数据清洗”。  
- 过滤器未通过仍合并数据。

---

## 8. 交付文档索引（数据侧）

| 文件 | 用途 |
| --- | --- |
| `MODEL_SIDE_HANDOFF_v1(1).md` | 模型侧收包与边界 |
| `AeroWF_v1_MODEL_TRAINING/MODEL_HANDOFF_v1.md` | 允许/禁止数据集 |
| `AeroWF_v1_MODEL_TRAINING/DATA_CONTRACT_v1.md` | 张量 · mask · 归一化 · 泄漏 |
| `AeroWF_v1_MODEL_TRAINING/RELEASE_NOTES_v1.md` | 规模 · sealed 路径 · stats |
| `release_v1/metadata/*.json` | 机器可读契约与清单 |
