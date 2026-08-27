# Closed-loop Auto Research 运行记录

本文按 ATM Agentic 框架图的 **核心闭环五步**（假设生成 → 研究动作 → 自动实验 → 独立评估 → 谱系记忆）记录一次真实 DeepSeek API 试跑。密钥只用进程环境变量，未写入 `.env`、配置或 `config.resolved.json`。

---

## 2026-08-25 · DeepSeek Chat × AeroWF ATC profile

| 项 | 值 |
| --- | --- |
| 时间 | 2026-08-25 13:25:50–13:26:38 UTC（约 48 s） |
| Provider | DeepSeek Chat Completions（`deepseek-chat`，`openai_compat`） |
| 配置 | `configs/atc.toml`；本跑将 `budget_per_axis` 临时设为 **1**（四轴各走完一轮五步闭环，便于对照框架图） |
| 流水线 | `examples/aerowf_research/pipeline`（persistence 基线） |
| 数据 | `AeroWF_v1_MODEL_TRAINING/release_v1`；搜索可见 `trainval/{train,val}`；机场 ZBAA / ZSPD / ZSSS |
| 样本 | `max_samples_per_split=128`；可见 val n=306 |
| 产物 | `runs/20260825T132550Z-ds-atc/`（`session.jsonl`、`trials/*.json`、`certification.json`、`summary.json`） |

### 总结果

闭环 **完整跑通**（四轴各 1 个 submitted trial，每 trial 都走完图中五步，失败也写入 lineage）。没有任何轴的 \(\overline{I}^{\mathrm{val}}\) 越过 \(\tau=0.005\)，因此 **没有冻结 \(c_a\)，held-out 认证表为空**——这是协议行为，不是中途崩溃。

| 轴 | trial | 五步是否走完 | 状态 | \(\overline{I}^{\mathrm{val}}\) | 独立评估结论 |
| --- | --- | --- | ---: | ---: | --- |
| representation | representation-000 | 是 | **no_gain** | +0.000 | 只改了 `features.py`；persistence 仍只用第 0 列，MAE 与基线完全相同 |
| model | model-000 | 是（实验抛错） | **failed** | — | 智能体手写 GRU，`fit` 签名与 harness 约定不一致 |
| physics | physics-000 | 是（实验抛错） | **failed** | — | 智能体按 DataFrame `.columns` 写物理项，AeroFrame 是 ndarray |
| data | data-000 | 是 | **no_gain** | +0.000 | 轴锁只允许 `data.py` / manifest；未声明 catalog 源，等于没增广 |

Baseline val（搜索可见，n=306）：MAE **1.149**，RMSE **1.630**，hazard CSI **1.0**；分机场 MAE ZBAA 1.251 / ZSPD 1.249 / ZSSS 0.946。

---

## 框架图五步：本跑每一圈具体做了什么

图中闭环对 **每一个 trial** 转一圈。本跑顺序：representation → model → physics → data。下面按五步写 **harness 做了什么** 和 **本 trial 智能体/实验实际产出**。

### 公共调度（每圈开始前）

Harness `ClosedLoopResearcher._one_trial`：从 pristine 复制流水线 → file-level ablation lock 只允许当前轴文件变化 → specialist 读 **model-visible lineage**（不含 test 标签、不含 `certification` 事件）。

---

### ① 假设生成 Hypothesis

**框架含义：** 针对当前轴提出一条可证伪假设（claim / mechanism / target_slice / expected_gain / falsification / negative_control），此时 **不写代码**。

**本跑：** 每次调用 DeepSeek，system 提示要求只输出 JSON 假设卡。四条假设都围绕「快速变化天气 / 对流」，但落点不同：

| trial | 轴 | 智能体 claim（摘要） | 预定证伪 |
| --- | --- | --- | --- |
| representation-000 | 表示 | `features.py` 把变量当静态独立特征，缺时空相关，动态天气 slice 差 | 加时空特征后目标 slice 提升不到 5% 则放弃 |
| model-000 | 模型 | persistence / 浅层聚合抓不住时序依赖 | 换成 GRU/LSTM/时序卷积后提升不到 5% 则放弃 |
| physics-000 | 物理 | 缺 CAPE / lifted index / 垂直风切变等过程量 | 加上后对流 slice 提升不到 5% 则放弃 |
| data-000 | 数据 | 缺测/错测处理差，高缺失 slice 差 | 改进缺失处理后提升不到 5% 则放弃 |

`session.jsonl` 事件：`hypothesis`（四次）。假设卡在提交实验前已冻结，后面评估按这张卡做 evidence gate。

---

### ② 研究动作 Research Action

**框架含义：** 在 **单一研究轴** 上把假设变成可执行改动（Data / Representation / Model / Physics），禁止跨轴。

**本跑：** 第二次 DeepSeek 调用，根据假设 + 当前轴允许文件快照生成干预。本跑全部选择 `kind=files`（未走 named preset）。

| trial | 允许改的文件 | 实际动作 | 轴锁 |
| --- | --- | --- | --- |
| representation-000 | 仅 `features.py` | 在 mask 池化上一时刻风/温/hour 上追加风、向、能见度、温度的时间差分，以及跑道间风速空间梯度 | `changed=['features.py']`，通过 |
| model-000 | 仅 `model.py` | 用 GRU + Adam、50 epoch MSE 替换 persistence；假定输入 `(n, seq_len, n_features)` | 只改 `model.py`，通过 |
| physics-000 | `physics.py` + `objective.py` | 在 `apply_physics` 里用 CAPE / LI / shear **占位代理**；notes 写明真实探空不可用 | 只提交了 `physics.py` |
| data-000 | `data.py` + `external_manifest.json` | 智能体判断缺测逻辑不在这两个文件里，**拒绝编造外部源**；`extra_source_ids()` 仍为 `[]`，manifest `sources: []` | `changed=['data.py','external_manifest.json']`（内容等价于空增广） |

`session.jsonl` 事件：`action`（文件正文不进 lineage，只记 kind/notes）。

---

### ③ 自动实验 Auto Experiment

**框架含义：** 在隔离工作区跑训练/推断，智能体不能改划分、指标或泄漏过滤器。

**本跑：** `run_trial` 把 pristine 拷到 `runs/.../trials/<id>/`，恢复非本轴文件，写入动作文件，再 `fit_predict(train, val)`。训练数据仅 `trainval/train`（+ data 轴经过滤器准入的 extras）。**val 标签给 evaluator，不给 specialist 当「改代码的依据」。**

| trial | 实验是否跑完 | 现场发生的事 |
| --- | --- | --- |
| representation-000 | 是 | 新 `featurize` 产出 8 列；`Model` 仍是 persistence，`predict` 只返回第 0 列（`prev_wind_speed`），后 7 列被丢掉 |
| model-000 | **否** | GRU `fit(self, X, y)` 两参数，harness 调用 `fit(X, y, sample_weight=...)` → `not enough values to unpack (expected 3, got 2)`；trial 目录随后删掉 |
| physics-000 | **否** | 生成代码访问 `X.columns`（把 ndarray 当 pandas）；`numpy.ndarray object has no attribute 'columns'`；trial 目录删掉 |
| data-000 | 是 | 无外部源可 merge，训练矩阵与基线相同 |

---

### ④ 独立评估 Independent Evaluation

**框架含义：** 评估器独立于智能体；搜索期 **只能打可见 val**；test / sealed / ZBAD 不可见。

**本跑：** `IndependentEvaluator.evaluate_workspace(..., split="val")`。端点：`overall.mae`、`ZBAA.mae`、`ZSPD.mae`、`ZSSS.mae`、`hazard.csi`。分数为相对冻结 persistence 基线的等权 \(\overline{I}\)。Evidence gate：目标提升 ≥ 0.005 且 CSI 不破 safety。

| trial | val MAE | CSI | \(\overline{I}\) | evidence |
| --- | ---: | ---: | ---: | --- |
| 基线 | 1.1486 | 1.0 | 0 | — |
| representation-000 | 1.1486（相同） | 1.0 | 0.000 | `no_gain`：预注册 slice 未过阈值；safety 通过 |
| model-000 | 无 | — | — | `failed`：实验未完成 |
| physics-000 | 无 | — | — | `failed`：实验未完成 |
| data-000 | 1.1486（相同） | 1.0 | 0.000 | `no_gain`：无增广，分数与基线相同 |

单轴归因可读：表示轴「改了特征、估计器不用」→ 零增益；模型/物理轴「代码级干预与合同不兼容」→ 失败进谱系；数据轴「没拿到可编辑的加载器 / 未声明 catalog 源」→ 零增益。

---

### ⑤ 研究谱系 / 经验记忆 Lineage & Memory

**框架含义：** 成功与失败都留下可复现记录，后续 trial 只读 **model-visible** 谱系。

**本跑写入：**

- `session.jsonl`：`run_start` → 每轴 `hypothesis` / `action` / `trial_result` → `freeze` → `certification`（路径，不含 test 分数）
- `trials/<id>.json`：完整假设卡、动作摘要、metrics 或错误 `notes`
- 成功跑通的隔离树：`trials/representation-000/`、`trials/data-000/`（含改过的轴文件）
- 失败 trial 的代码树已删除，错误留在 json `notes`

后一轴的假设调用能看到前几轴的 `trial_result`（例如 model 仍写「动态天气」，与 representation 的 claim 同主题），但 **看不到** held-out 标签。

`freeze` 的 `selected` 为 `{}`：没有轴满足 \(c_a\) 选择阈值，故跳过「冻结最佳方案 → held-out」里的配置重训。`certification.json` 仍写出协议头、\(\tau=0.005\) 和 **仅基线** 的 temporal / spatial / event 分数（认证器对基线工作区打了三类 hidden split），`axes` 与 `routed` 的选轴为空。

---

## 认证侧（框架图右侧，搜索结束后）

因无 \(c_a\)，**没有**「冻结智能体方案再评一次 hidden test」。基线在 evaluator-owned 分区上的分数（供对照，**未**用于搜索选模）：

密封包未挂载：temporal = `trainval/val` 时间尾部 20%（seed 42）；spatial = ZBAD 契约气候探针。

不要把本次空 `axes` 写成「智能体方案泛化失败」——选模阶段就没有可冻结的改进。

---

## 与框架图 / 论文的对应关系

| 框架图 | 本跑 |
| --- | --- |
| 研究起点：问题 + AeroWF 基线 + 多源数据 | 机场天气预报；persistence + DATA_CONTRACT；trainval + 可选 pretrain catalog |
| 四轴动作空间 | 四轴各 1 trial，锁生效 |
| 避免 val 过拟合 | 搜索只打 val；test 不进 lineage |
| 识别 distribution shift | 本跑没有可认证的 \(c_a\)，shift 签名未触发 |
| 可泛化发现 | 本次未产生（零正增益轴） |

智能体更愿手写 GRU / pandas 物理项，而不是 `ridge` / `extreme_wind_weights` 等合同预设，所以 model/physics 在实验步失败。这与 2026-08-21 dummy API 试跑类似：假设卡完整，代码级干预容易与 evaluator 合同错位。

---

## 复原说明

- 密钥未写入仓库；`config.resolved.json` 中 `api_key` 为空。
- 试跑产物保留在 `runs/20260825T132550Z-ds-atc/`，未删。
- 一次性启动脚本已删除；`__pycache__` 已清理。

---

## 2026-08-21 · Dummy profile × DeepSeek Chat（历史）

当时配置为 dummy 合成气象、单一 MAE Δ、机场 ZBAA/ZSPD/ZGGG。**不要**与上表 AeroWF \(I_t\) 混报。

四轴里当时只有 **model** 越过阈值（val ΔMAE +0.840），并在 temporal / spatial / event 上保持正迁移。representation 访问了不存在的 `WeatherFrame.columns`；data 提交了未知源 `metar_subhourly`。详见当时 `runs/api-run/`（已删除）。
