# 轴 2 · Model（对应论文 Model Axis）

> 论文轴名：`model`  
> AeroWF 轴名：`model`  
> 协议总览：[00-protocol-reproduction.md](00-protocol-reproduction.md)

---

## 1. 论文定义（严格）

Model 轴 **只改变估计器、优化与正则**，表示与训练证据保持 pristine：

- 可编辑：估计器家族、容量、超参、概率校准、多 seed 集成等（附录 S5：modl / calib）。
- 不可编辑：特征代码、外部数据写入、泄漏过滤器、划分、指标实现。
- 归因：ablation lock 恢复非 model 文件；一次 trial 仅 model 表面可变。

Matched-trial AutoML 对照（论文贡献 4）：在 **相同特征、相同 trial 数** 下，标准超参搜索不得复现 agent 的 **代码级** 结构干预。AeroWF 复现时必须保留该对照位。

报告单位：验证集上 aggregate \(I\) 最大的单一快照 \(c_{\mathrm{model}}\)，再 held-out 认证一次。

非迁移警告（论文 Table 1）：TDC model 轴 val **0.041** → test **+0.003**（selection variance 典型）。**禁止**只报验证增益。

---

## 2. AeroWF 映射

| 论文概念 | AeroWF 落地 |
| --- | --- |
| CatBoost / 估计器家族 | AeroWF 网络模块、头、聚合器、融合方式 |
| 优化与正则 | 优化器、调度、dropout、weight decay、梯度裁剪 |
| 概率校准 | 分位数头 / CRPS / 校准层（若本轴允许改头；损失权重属 physics 轴则禁止） |
| 强紧凑基线 | 冻结 AeroWF checkpoint 拓扑的“未编辑”版本 |

### 2.1 与数据接口的边界

本轴 **消费** `任务测试/AeroWF/数据接口` 中的训练侧交付，但：

- 只读 `pretrain/train|val`、`trainval/train|val`；  
- 不改 `DATA_CONTRACT_v1` 通道顺序；  
- 必须尊重 `runway_mask`（padding 不得进 attention / pooling / 重建损失）；  
- 归一化只用冻结 JSON，禁止在 model trial 中 refit。

### 2.2 文件级锁

| 允许修改 | 禁止修改 |
| --- | --- |
| `model.py`（及明确列入 model 轴的网络模块文件清单） | `features.py`, `physics.py`, `objective.py`, `data.py`, `external_manifest.json`, evaluator |

若一次改动同时触及损失权重与结构：按论文精神应拆成 **两个** 单轴 trial（结构 → model；损失 → physics），否则归因失效。

### 2.3 当前代码

首版 ATC 基线是 **紧凑 CPU persistence**（`examples/aerowf_research/pipeline/model.py`），对应论文「强、未逐任务调参」的起点，而不是已训练的 3D 网络。预设 `ridge` 把估计器换成 `sklearn.linear_model.Ridge`，只改 `model.py`。

pytest / offline 闭环里，model 轴是唯一稳定越过 \(\tau=0.005\) 并进入 held-out 认证的轴。Matched-trial AutoML（FLAML）对照位保留在合同中，代码尚未接入。

---

## 3. Trial 合同

假设卡必填字段同总协议；model 轴额外要求：

| 字段 | 要求 |
| --- | --- |
| mechanism | 结构/容量/校准如何针对已诊断 failure slice |
| expected_gain | 对哪些端点 / 时距 / CSI 提升 |
| falsification | 结构改动未触及目标 slice，或 safety 指标越界 |
| negative_control | 在“机制不应作用”的 slice 上不得同等提升 |

执行：

1. lineage → 假设  
2. 只改 model 表面  
3. 低保真 smoke（可选）→ 完整可见 val（建议 3–5 seed，论文强调小验证集上的选择噪声）  
4. 改进则存 snapshot；失败也写 lineage  
5. 轴内串行，等预算

搜索奖励 **仅** 来自 `trainval/val`（及允许的 `pretrain/val` 选型），永不来自 sealed。

---

## 4. 评分、冻结与认证

1. \(c_{\mathrm{model}}=\arg\max_c \overline{I}^{\mathrm{val}}(c)\)。  
2. 冻结模型代码 + pristine 表示/数据/物理。  
3. 在 `trainval/train`（及既定预训练设定）上重训。  
4. 对 temporal / spatial / event 各评 **一次**（官方 sealed 优先，见总协议 §4）。  
5. 报告 \(R^{\mathrm{val}}\)、\(R^{\mathrm{test}}\)、generalization gap \(\Gamma=R^{\mathrm{val}}-R^{\mathrm{test}}\)，以及 `signature`。

Routed：仅当该轴验证回报 \(\ge 0.005\) 时可被端点选中。

---

## 5. 首批可检验假设（示例）

1. **时距条件融合**：horizon-conditioned fusion，预期改善长时距，负对照 = 短时距不应显著回退。  
2. **跨跑道注意力**：用 mask-aware attention 替代简单聚合，目标 = 多跑道机场关联误差。  
3. **概率预测头**：输出分位数，用 CRPS/coverage 评价（若损失改动进 physics 轴，则本轴只改头结构、固定默认损失）。  
4. **容量-数据匹配**：按样本量调节宽度/深度，避免小样本过拟合（对照：同预算 AutoML 超参搜索）。

---

## 6. 禁止事项

- 本轴 trial 改特征工程或引入外部机场/年份数据。  
- 根据 sealed 分数改结构后再次提交（破坏 one-shot certification）。  
- 把 AutoML 超参搜索结果冒充“代码级 model 发现”而不做 matched-trial 对照。  
- 忽略 `runway_mask`，把 padding 当真实跑道。
