# 轴 3 · Physics / Domain（论文三轴之外的 ATC 扩展轴）

> 论文原文轴集：`feature | model | data`（无独立 physics 轴）  
> ATC / 导师方案：第四轴 **Physics / Domain / Objective**  
> AeroWF 轴名：`physics`  
> 协议总览：[00-protocol-reproduction.md](00-protocol-reproduction.md)

本轴 **严格沿用论文的单轴归因与 held-out 认证规程**，仅把“可编辑表面”定义为：**气象机理与空管领域约束注入、预训练/下游目标函数与样本加权**，而表示代码、模型结构文件、外部证据入口保持 pristine。

---

## 1. 为何单独成轴（归因需要）

若把损失与结构写在同一次 trial，则无法回答论文的核心问题：

> 增益来自哪一类科研动作？该动作是否在 held-out 上迁移？

因此：

- 改 DTW/频谱对齐权重、极端天气加权、CRPS、物理一致性惩罚、跨任务梯度冲突缓解 → **physics**  
- 改网络拓扑、注意力、头结构 → **model**  
- 改输入通道派生 → **representation**  
- 改训练样本来源 / 外部证据 → **data**

File-level lock 强制互斥。

---

## 2. AeroWF 映射

| 允许动作 | 说明 |
| --- | --- |
| 预训练目标 | 几何对齐、重建、频谱对齐；**时距条件化权重**（回应 Scratch 在 T+1 优于预训练的现象） |
| 下游损失 | MAE/MSE 加权、分位数/CRPS/NLL、危险天气加权 |
| 物理一致性 | 风分量约束、气压-风耦合惩罚、跑道坐标系物理量一致性 |
| 领域安全约束 | CSI/POD 不回退、FAR 上限，作为 **硬门槛**（论文式 reliability，而非可交易 soft loss） |
| 样本加权 | 极端风 / 低能见度 / 传感器缺失重采样权重 |

### 2.1 与数据契约的关系

- 不修改 `runway.npy` 通道顺序与 `DATA_CONTRACT_v1`。  
- SSL：`weather_label.npy` **不得**作为自监督目标（`MODEL_HANDOFF_v1.md` §4）。  
- 统计量仍只来自 train-only 冻结 JSON。  
- 物理项若需要额外中间量，必须在 **train/val 可见数据** 上由现有通道计算，不得偷看 sealed。

### 2.2 文件级锁

| 允许修改 | 禁止修改 |
| --- | --- |
| `physics.py`, `objective.py` | `features.py`, `model.py`, `data.py`, `external_manifest.json`, evaluator |

### 2.3 当前代码

| 文件 | 基线行为 |
| --- | --- |
| `physics.py` | `apply_physics` 恒等 |
| `objective.py` | 均匀 `sample_weights` |
| 预设 `extreme_wind_weights` | 对 \(y \ge\) hazard 阈值的样本权重 ×5 |

Evidence gate：`research/evidence.py`。CSI 相对基线回退超过 `safety_csi_tolerance`（默认 0.02）→ `unsafe`，不进入 \(c_a\)。persistence 忽略权重时 physics trial 为 `no_gain`，归因上表示「本轴未改估计器」。

---

## 3. Trial 合同（可证伪优先）

Physics 轴假设必须是 **机制可证伪** 的，而不是“换个损失试试”：

| 字段 | 示例 |
| --- | --- |
| claim | 统一几何预训练过度平滑短时风场 |
| mechanism | 长时距 DTW 权重主导梯度，抑制 T+1 高频 |
| target_slice | T+1 MAE / 突发风 slice |
| expected_gain | 降低 T+1 的 DTW 权重应改善 T+1，T+8 不明显变差 |
| falsification | 改善仅出现在单机场，或 CSI 显著下降 |
| negative_control | 均匀权重不应单独抬高 CSI |

Evidence gate（对齐导师 SCAR 方案）：

\[
\text{Claim Supported} \iff
\text{Predicted Slice Improved}
\land \text{Negative Control Passed}
\land \text{No Safety Violation}
\]

未通过则 trial 不得标为成功发现，必须降级结论或丢弃。

---

## 4. 评分与安全门槛

论文用单一 \(I_t\) 聚合；气象安全任务增加 **约束优先**：

硬门槛（不满足 → `unsafe`，不进入 \(c_a\) 竞争）：

- 危险天气 CSI / POD 相对基线回退不超过 \(\epsilon\)  
- FAR 不超过业务阈值  
- （若有）校准覆盖率落在允许区间  
- 推理时延 / 显存不超过部署预算  

通过门槛后，再用归一化 \(I_t\) / robust 分数排序。选择 \(c_{\mathrm{physics}}\) 与认证规程同论文 §3.1–§3.3。

---

## 5. 首批可检验假设（示例）

1. **时距条件化预训练**：随 horizon 调整 DTW/频谱/重建权重。  
2. **极端天气加权**：高风速 / 低能见样本 upweight，目标 CSI↑，负对照 = 均匀权重。  
3. **CRPS / 分位数损失**：在固定模型头（或与 model 轴分两次 trial）下改善校准。  
4. **物理一致性惩罚**：风矢量重建与气压倾向一致性，检验是否减少非物理预报。

---

## 6. 禁止事项

- 借“物理约束”之名改网络结构或偷偷加特征通道。  
- 用 sealed 事件标签调损失权重。  
- 用平均 MAE 掩盖 CSI 崩塌。  
- 把 physics 与 model 混在同一 diff 中提交。
