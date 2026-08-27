# 接下来要做什么

> 依据：2026-08-25 DeepSeek × AeroWF 试跑（[API运行结果.md](../API运行结果.md)）、[arXiv:2606.22731](https://arxiv.org/pdf/2606.22731)、ATM Agentic 框架图（假设 → 动作 → 实验 → 独立评估 → 谱系，再 **冻结 + held-out 认证**）。  
> 目标：从「五步能转完」推进到「能发现可归因改进，并认证是否泛化」。

当前 harness 已经具备论文的骨架（单轴锁、独立 evaluator、\(I_t\)、certify-after-search、AeroWF 数据边界）。缺口不在「再画一张流程图」，而在 **动作合同太松、谱系没有真正驱动下一圈假设、认证侧没有可冻结的 \(c_a\)**。

---

## 1. 本跑暴露的不足（对照框架图五步）

| 步骤 | 框架要求 | 本跑事实 | 后果 |
| --- | --- | --- | --- |
| ① 假设 | 可证伪、针对本轴 failure slice | 四轴都写成「动态/对流天气」，与 persistence 基线和 DATA_CONTRACT 通道对不齐 | 假设无法被实验证伪成「机制成立/不成立」 |
| ② 动作 | 单轴、可执行、与 evaluator 合同一致 | 全部手写 `files`，不用 preset；GRU / pandas `.columns` | model、physics 在实验步崩溃 |
| ③ 实验 | 隔离跑通 | 失败 trial 目录被删，只留一句 `notes` | ⑤ 几乎学不到「怎么改才能跑」 |
| ④ 评估 | 独立 val；端点 \(I_t\) | 表示轴改了 8 列，persistence 只用第 0 列 → \(I=0\) | 表示轴永远像「没干活」 |
| ⑤ 谱系 | 用实测更新后续假设 | 后一轴仍重复「动态天气」主题；失败原因未结构化进 prompt | 闭环在 ④→⑤→① 处断开 |
| 右侧认证 | 冻结 \(c_a\)，held-out 只评一次 | `selected={}`，`axes` 空 | 无法谈 transfer / selection variance / distribution shift |

论文对照：MapLight 式 **强紧凑基线 + 代码级但合同内的干预**；本跑基线是 persistence，智能体却直接上 GRU，既不是「在强基线上做轴归因」，也没有活到认证。

---

## 2. 优先顺序（建议按此做）

### P0 · 让至少一条轴能产生可冻结的 \(c_a\)（否则认证侧永远空）

没有 \(c_a\)，框架图右侧整段无法验收。

1. **估计器合同（修 ②③，专治 model-000）**  
   - 固定 `Model.fit(X, y, sample_weight=None)` / `predict(X) → (N,)`。  
   - 动作 prompt 强制：优先 `preset=ridge`；手写代码必须遵守该签名；禁止假定 `(N, T, C)` 除非 featurize 真的输出序列。  
   - 实验失败时把 traceback **结构化**写进 lineage（保留失败快照或至少保留 `model.py` 全文），供下一圈 ① 阅读。

2. **表示轴可被估计器看见（修 ④，专治 representation-000）**  
   - persistence 只用第 0 列时，表示轴 \(I\) 恒为 0，论文式 feature 归因不成立。  
   - 做法二选一（不要混在一次 trial）：  
     - A. 基线改成会使用全部列的紧凑估计器（Ridge/CatBoost，对齐论文 MapLight+CatBoost）；表示轴只改 `features.py`。  
     - B. 保持 persistence 作「弱对照」，另设 `baseline = ridge_on_frozen_features` 作为论文意义上的强基线。  
   - 推荐 **A**：与 2606.22731 §3.6「强、未逐任务调参」一致，也才能问「表示还值不值得改」。

3. **物理轴张量合同（修 ②③，专治 physics-000）**  
   - prompt / 预设写明：`X` 是 `ndarray`，`frame.runway` 形状 `(N,4,96,11)`，用 `runway_mask`，**没有** `.columns`。  
   - 预设 `extreme_wind_weights` 应作为 physics 默认动作；CAPE 等无探空时禁止用假列冒充机理。

4. **数据轴可执行增广（修 ②，专治 data-000）**  
   - 假设卡禁止「改加载器里的缺失插补」（加载器在锁外）。  
   - 只允许 catalog：`pretrain_train` / `shifted_climate` / `leak_val`（后者必须被 L2 拒绝，作为过滤器回归）。  
   - prompt 明确：`write_external_data` + `extra_source_ids()` 就是本轴的全部动作空间。

**P0 验收：** 同等预算下（先 1 trial/轴，再 30/轴对齐论文迁移研究）offline 或 DeepSeek 至少让 **model 轴** `improved` 并写出非空 `certification.axes.model` 的成对 val/test。

---

### P1 · 接上 ④→⑤→①（框架图真正的闭环）

论文：「later agents see this record and can retain, revise, or abandon the direction。」本跑没有 revise。

1. **Lineage 编译进假设 prompt**  
   - 不只 dump 最近 JSON，而要短表：轴、状态、`I`、错误类型（`FitSignatureError` / `FrameApiError` / `NoGainPersistenceIgnoresFeatures` / `EmptyManifest`）。  
   - 规则：同一失败模式不得原样再提；`no_gain` 必须改机制或放弃该方向。

2. **轴内串行、跨轴只读谱系**  
   - 论文：轴内 trial 一个接一个。`budget_per_axis=1` 时跨轴应明确「表示失败 ≠ 换 GRU」，而是「表示未被估计器使用」。  
   - 可选 meta specialist（论文 S5）：读四轴 lineage 后只建议下一轴方向，不改代码。

3. **Evidence gate 对得上假设卡**  
   - 本跑 `target_slice` 是「高变率天气」，评估却用全局 \(\overline{I}\)。  
   - 下一步：evaluator 计算假设声明的 slice（例如风暴 `event_id` / 高风速）上的 \(I\)，负对照 slice 不得同等提升。未算 slice 则不得标 `supported`。

**P1 验收：** 第二圈假设的 `parent_trial_id` 非空，且 claim 引用上一圈 `notes` 或错误类型；不再出现四轴同一套「对流」套话。

---

### P1 · 认证与泛化（框架图右侧，论文 §3.3 / §5）

只有 P0 产生 \(c_a\) 之后才有意义。

1. **挂官方 sealed**  
   - 训练包无 `sealed/**`。把评测侧指到 `domain.sealed_root`：`sealed/temporal`、`sealed/spatial/ZBAD`。  
   - Fallback（val 时间尾部 + 合成 ZBAD）只能标 `probe`，不得当论文主表。

2. **成对报告与两种非迁移签名**  
   - 主表：每轴 \(R^{\mathrm{val}}\) | \(R^{\mathrm{test}}\)，routed \(\tau=0.005\)。  
   - val 大、test≈0 → selection variance（论文 TDC model）。  
   - val 正、test 负 → distribution shift（论文 Polaris data）。  
   - 等预算 ≥30 trial/轴后再谈签名；1 trial 不够。

3. **Data 轴审计表**  
   - 对齐论文 Table S7：源、overlap、L1–L4 裁决、是否准入。过滤器通过 ≠ 分布匹配。

**P1 验收：** 一次 DeepSeek 跑（≥30 trial/轴或至少 model 轴 30）产出非空 routed held-out，且 session 中无 test 标签。

---

### P2 · 对齐论文其余贡献 / 框架图「通用 ATM 科研」

| 项 | 论文 / 框架 | 现状 | 下一步 |
| --- | --- | --- | --- |
| 强基线 | MapLight + CatBoost | persistence | CatBoost/LightGBM 紧凑基线，无逐任务调参 |
| 等预算 | TDC 100/轴；迁移 30/轴 | 试跑 1/轴 | `atc.toml` 保持 30；禁止 online 重分配 |
| AutoML 对照 | FLAML matched-trial | 未做 | 同特征、同 trial 数、同 split，只搜超参 |
| 预训练对照 | Uni-Mol 共享 train split | 未做 | 可选：Aero-Pre2020 SSL 后下游微调，**不得**用 `weather_label` 当 SSL 目标 |
| 真实 `.npy` | DATA_CONTRACT 张量 | index 物化 | 解压训练 7z 后走真实 `runway.npy` |
| 多任务外推 | 流量/冲突/运行优化 | 仅天气 nowcast 代理 | 端点合同稳定后再加任务 adapter |
| 安全门槛 | CSI/POD/FAR | CSI 已有，本跑饱和为 1.0 | 用物理阈值定义危险天气，避免 CSI=1 无信息 |

---

## 3. 建议的实施切片（可直接开干）

| 顺序 | 切片 | 主要文件 | 完成定义 |
| --- | --- | --- | --- |
| 1 | Model 合同 + 失败谱系 | `specialist.py`、`experiment.py`、`evaluator.py` | GRU 式签名错误进入 lineage；ridge preset 为默认 |
| 2 | 强基线改 Ridge/CatBoost | `examples/aerowf_research/pipeline/model.py` | 表示轴改列后 val \(I\) 可以非零 |
| 3 | Physics/Data prompt 与 catalog | `presets.py`、`offline.py`、动作 system prompt | 不再生成 `.columns`；data 默认 `pretrain_train` |
| 4 | Slice-aware evidence | `evidence.py`、`metrics.py` | 假设卡的 target_slice 有对应 mask |
| 5 | Lineage → 下一假设 | `loop.py`、`session.py` | ⑤ 编译错误类型；① 强制引用 |
| 6 | 正式 API 复跑 | `configs/atc.toml` | 至少 model 轴进入认证；更新 [API运行结果.md](../API运行结果.md) |
| 7 | sealed + 30 trial | 数据接口 + 预算 | 成对 val/test 与签名可报告 |

不要在 1–3 完成前加长预算或换更大模型：预算只会放大「写不跑的代码」。

---

## 4. 明确不做什么（避免稀释论文问题）

- 不把 val 分数或 sealed 标签喂回假设生成。  
- 不把 representation + model 写进同一次 trial（破坏单轴归因）。  
- 不把 index 物化或合成 ZBAD 的分数写成官方 AeroWF leaderboard。  
- 不把「换 GRU」当成 AutoML 对照；对照必须是固定估计器族上的超参搜索。  
- 不把 CSI=1.0 的饱和 val 解释成危险天气已解决。

---

## 5. 一句话

框架图的五步 **调度已经通**；论文要的「发现并认证可泛化改进」还没开始。下一步是把动作空间收成 **可执行合同**，让谱系真正改写下一圈假设，直到出现可冻结的 \(c_a\)，再谈 held-out 与跨机场/极端天气迁移。
