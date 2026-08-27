# 轴 1 · Representation（对应论文 Feature Axis）

> 论文轴名：`feature`（§3.1，\(A=\{\mathrm{feature},\mathrm{model},\mathrm{data}\}\)）  
> AeroWF 轴名：`representation`  
> 协议总览：[00-protocol-reproduction.md](00-protocol-reproduction.md)

---

## 1. 论文定义（严格）

Feature 轴 **只改变分子表示**，估计器与训练证据保持 pristine：

- 可编辑：物理化学描述符、子结构 / SMARTS alerts、文献 grounding 特征（附录 S5：fphs / fsub / lit）。
- 不可编辑：模型家族、优化器、正则、外部数据合并逻辑、evaluator、划分、指标。
- 归因：file-level ablation lock 在每次 trial 前从 pristine 恢复非 feature 文件；一次 trial 只允许 feature 表面变更。

报告单位：该轴全部 trial 中，验证集平均归一化提升最大的 **单个** 快照 \(c_{\mathrm{feature}}\)，再做 held-out 一次性认证。

---

## 2. AeroWF 映射

| 论文概念 | AeroWF 落地 |
| --- | --- |
| 分子指纹 / 描述符 | 跑道张量通道变换、外生特征派生、编码方式 |
| 子结构 alerts | METAR 天气码 / 危险天气离散编码增强 |
| 文献特征 | 领域知识驱动的可计算特征（须可复现、可进 diff） |
| MapLight 固定表示 | `DATA_CONTRACT_v1` 冻结通道顺序 + 基线 featurizer |

### 2.1 张量契约（只读，不可由本轴改契约）

数据侧契约来自 `任务测试/AeroWF/数据接口/.../DATA_CONTRACT_v1.md`：

- `runway.npy`：`(N, 4, 96, 11)`，通道顺序冻结：  
  `cloud_base, wind_x, wind_y, pressure, temperature, humidity, dewpoint, hour_sin, hour_cos, month_sin, month_cos`
- `runway_mask.npy`：`(N, 4)`，`False` = padding，**禁止**用数值推断跑道有效性
- `exo_continuous.npy`：`(N, 3)` → `visibility, cloud_height(log1p), gust_speed`
- 类别 / 二元：`weather_code_id, sky_condition, has_gust, is_cavok`（不得对类别做连续 Z-score）

本轴可以 **在契约之上** 派生新表示（例如跑道坐标系下的顺风/侧风、露点差、阵风变化率、多尺度频谱能量），但：

1. 不得改动交付包中的原始 `.npy` 布局与通道序号约定；  
2. 不得把 sealed / ZBAD / test 样本纳入表示选择或统计量拟合；  
3. 归一化必须复用 `train_stats_v1.json` / `pretrain_stats_v1.json`，禁止 refit。

### 2.2 文件级锁（harness）

| 允许修改 | 禁止修改 |
| --- | --- |
| `features.py` | `model.py`, `physics.py`, `objective.py`, `data.py`, `external_manifest.json`, `pipeline.py`, evaluator, splits |

提交前 `assert_axis_edits`：越界直接拒绝。

### 2.3 当前代码

| 项 | 位置 |
| --- | --- |
| 基线 featurizer | `examples/aerowf_research/pipeline/features.py`：mask 池化上一时刻风速、温度、hour |
| 预设 `runway_wind` | `research/presets.py` `AEROWF_PRESETS`：追加 `wind_x` / `wind_y` / 合成风速 |
| dummy 预设 | 由 `wind_dir` 与跑道朝向算 headwind / crosswind |
| 锁 | `axis_lock.py` → 仅 `features.py` |

若基线估计器是 persistence（只用第 0 列），仅改表示 **不会** 改变 val 分数。这是 ablation lock 生效，不是 bug。要验证表示轴，需与会使用全部列的 model 配置分开报告，或等 model 轴先换成可学习估计器后再做 representation trial（两次单轴，禁止一次 diff 混改）。

---

## 3. Trial 合同

每个 representation trial 必须先提交结构化假设卡，再允许改代码：

| 字段 | 要求 |
| --- | --- |
| claim | 表示缺陷的可检验陈述 |
| mechanism | 为何当前通道/编码导致该失败 |
| target_slice | 机场 / 时距 / 天气类型 / 传感器缺失等 |
| expected_gain | 相对基线的归一化提升方向与量级 |
| falsification | 何种结果证伪该机制 |
| negative_control | 无关 slice 上不应出现同等增益 |

流程（论文 §3.4）：

1. 读 lineage  
2. 提出假设（不写代码）  
3. 只改 `features.py`  
4. 提交 → 独立 evaluator 在 **可见 val** 上打分  
5. 写入 lineage（成功与失败均保留）

搜索期 **禁止** 请求 `sealed/temporal`、`sealed/spatial`、`pretrain/test` 标签。

---

## 4. 评分与选择

- 指标：相对冻结基线的 \(I_t\)（`overall.mae` / `{ICAO}.mae` ↓，`hazard.csi` ↑）。  
- 选择：\(c_{\mathrm{repr}} = \arg\max_c \overline{I}^{\mathrm{val}}(c)\)。  
- 认证：冻结 featurizer + 其余 pristine 文件，重训，对 temporal / spatial / event 各评一次。  
- 成对报告：\(R^{\mathrm{val}}\) 与 \(R^{\mathrm{test}}\)；若 val 升、test 塌，记为 selection variance 或 shift（见总协议 §2.7）。

论文参考：MoleculeNet 上 feature 轴 held-out 为正（Table 1）；TDC feature 近饱和（val 0.013 → test +0.001）。AeroWF 上需单独测量，不得预设“表示轴必赢”。

---

## 5. 首批可检验假设（示例，须预注册）

1. **跑道相对风**：由 `wind_x/wind_y` + 跑道朝向派生 headwind / crosswind / gust-crosswind，预期提升短时距运行风误差，负对照为仅改 hour 编码。  
2. **气压/露点趋势**：在 96 步窗内加入倾向特征，目标 slice = 急压降后阵风；负对照 = 气压稳定窗。  
3. **完整 METAR 天气码**：强化 `weather_code_id` 嵌入，目标 = 危险天气 CSI；不得以降平均 MAE 换 CSI 崩塌（safety gate）。  
4. **多尺度频谱能量**：STFT/Wavelet 能量作为附加通道；检验是否改善突发风而非仅日周期。

---

## 6. 禁止事项

- 在本轴 trial 中改网络结构、损失、学习率、外部数据源。  
- 用 sealed 分数做特征筛选。  
- 改通道顺序或破坏 `runway_mask` 语义。  
- 对 PRE2020 把 `weather_label.npy` 当 SSL 目标（与表示无关，但同属契约红线）。
