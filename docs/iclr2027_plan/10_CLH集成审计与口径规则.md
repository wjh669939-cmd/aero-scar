# CLH（closed loop research）集成审计与指标口径规则 v1.1（补写版）

> 日期：2026-08-25 首版，2026-08-26 凌晨补写（v1.0 文件遗失，本版 = 原审计 + 状态更新 + P0 补丁规格 + kernel 处置）
> 审计对象：`/root/autodl-tmp/closed loop research/`（约 1,000 行核心代码 + 8/25 DeepSeek API 试跑记录）
> 结论一句话：**协议骨架合格、停止打磨；离正式 discovery 差 1 个必修漏洞 + 5 个集成缺口（约 4–6 人天），其中 3 项已于 8/26 凌晨完成。**

---

## 一、审计确认的成果（不用再动的部分）

1. 五步闭环（假设→动作→实验→独立评估→谱系）用真实 DeepSeek API 端到端跑通，四轴各 1 trial；
2. file-level axis lock 生效（`assert_axis_edits` 实测拦截越界）；
3. 失败如实进谱系：no_gain / failed 区分，无正增益时不冻结、不认证——协议行为正确；
4. 认证分数不进 `SessionLog.model_visible()`；
5. agent 拒绝编造外部数据源（data-000 交空 manifest），提示词的诚实约束有效。

**判定：闭环逻辑本身不需要新功能。以下全部是接线与配置，不是重构。运行时决策见 11 文档：radar harness 退役，CLH 为唯一 discovery 运行时。**

## 二、必修漏洞（P0，阻塞挂载封存数据）——**已修复（8/26 凌晨）**

**原问题**：`evaluator.py` 的 `_TrialImport` 经 `importlib` 把 trial 的 `pipeline.py` 加载进 evaluator 自身进程。sealed_root 挂载后，agent 代码与持有隐藏数据访问权的进程同权——custody 主张即破。

### 实施结果（对照原规格）

已在 CLH 中落地，新增 `src/clh/research/subproc.py` + `src/clh/research/trial_runner.py`，重写 `evaluator.py`：

1. ✅ trial 经 `subprocess.run([sys.executable, "-m", "clh.research.trial_runner", ...])` 独立子进程执行；环境变量**白名单**（仅 PATH/HOME(tmp)/PYTHONPATH(clh src)），API key、封存路径变量一律不透传（有断言测试）；
2. ✅ 数据以序列化 npz 只读文件交给子进程（`allow_pickle=False`）；子进程只做 加载 → `fit_predict` → 写 `predictions.npy`；超时/非零退出 → `EvaluatorError` → trial 记 `failed`；
3. ✅ evaluator 进程只读预测文件评分，**永不 import** trial 代码（`_TrialImport` 已删除）；
4. ⚠️ 与原规格的一处偏差：data 轴解析路径**未删除**而是移入同一隔离子进程（`--mode extras` 探针）——原因是 dummy 协议测试仍覆盖 data 轴，且 custody 性质等价（evaluator 进程仍不执行 agent 代码）；atc 配置中 data 轴保持禁用；
5. ✅ 验收：新增 `tests/test_subproc.py` 6 项测试，含**标签回显作弊测试**与**密钥泄漏测试**；全套 19 项测试通过。

### 修复中顺带封堵的新漏洞

原进程内实现把带标签的 `eval_frame` 整个交给 trial 的 `fit_predict`，agent 代码理论上可 `return eval_frame.y` 直接刷满 val 分、污染整个搜索。现在 eval 帧序列化时剥离 `y/hazard/weather_label` 后才进入子进程；作弊 pipeline 会因属性不存在直接崩溃（有测试固定此行为）。train 帧标签照常提供（训练合法所需）。

## 三、五个集成缺口与状态（8/26 凌晨更新）

| # | 缺口 | 原始问题 | 要求 | 状态 |
|---|---|---|---|---|
| G1 | 执行器载荷 | `fit_predict` 内存内运行、180s 超时、persistence 基线 | 异步 GPU 作业：真实 AeroWF 训练、seed 注入、checkpoint+digest 登记、按 trial 记账 | 未动，依赖 D 仓库布局（8/27） |
| G2 | 评测口径 | τ=0.005、等权 Ī 照搬分子论文；试跑 baseline CSI=1.0 为退化值 | 换 C 的冻结 evaluator + DecisionPolicy；τ 与等权口径废弃 | **要求已下发**（12 文档），C 8/29 冻结 |
| G3 | 轴配置 | 四轴 representation/model/physics/data 照搬分子论文 | 对齐冻结注册表：R / O_tier1（+条件 M、条件 O_tier2）；physics 并入 R；data 禁用 | **已完成（8/26）**：CLH 新增独立 `objective` 轴（锁 `objective.py`），`configs/atc.toml` 改为 `axes=["representation","objective"]`；M 轴若 DEC-001 批准，加回 `"model"` 一行即可 |
| G4 | 随机对照臂 | 无 | registry active 模板均匀合法采样，镜像 trial 数 | **已实现并测试**：`aerowf-v1/tools/contract_tools/random_arm.py` |
| G5 | 接口合同进提示词 | 试跑 2/4 trial 死于接口错位（fit 签名、ndarray 当 DataFrame） | system prompt 内嵌 `fit_predict` 合同；smoke 失败给一轮修复（修复轮不另计 trial 但入账本） | 未动，与 A-4 提示词工程一起做 |

A 侧已完成件（28 项单测全绿，正典位置 `workspace/runs/aerowf-v1/tools/contract_tools/`；8/26 凌晨已合并此前误建在退役仓库中的并行实现，副本已删，详见 09 文档 §二）：

- **A-1 schema 校验器** `validate.py`：trial/result 合规校验 + M 轴参数预算 gate（±5% 容量混淆拒绝）；
- **A-2 axis-lock 引擎** `axis_lock.py`：越界拒绝、非活跃轴拒绝、空 diff 拒绝、隐藏数据路径 token 永拒；`<AEROWF_REPO>` 占位路径落定后即可上岗；
- **A-3 随机臂采样器** `random_arm.py`：seed 可复现、只采 active 模板、param_space 参数采样（registry 已补 9 个动作）、输出直接通过 schema 校验；
- **A-4 LLM 提案解析器** `proposal_parser.py` + `prompts/llm_proposal.md`：严格 JSON、科学字段无默认值、自由提案标记、模板假设须复述；
- **A-6 evaluator 客户端** `evaluator_client.py`：对齐 12 文档 CLI 合同，三态语义，超时/崩溃/坏 JSON 全归 failed 不误判候选。

**部署位置待定（阻塞项，8/27 站会定）**：CLH 引用 `../AeroWF/数据接口/release_v1`，本机不存在。二选一：同步数据包到本机 / discovery 在组员或租卡机器执行。

## 四、指标口径规则（即日生效，写给全组）

背景：本地与论文 MSE 出现 ~76 倍差异；且早前"本地 1epoch val MSE 0.23433 ≈ 论文 0.236"与其矛盾。

1. **撤回一个推论**："0.23433 ≈ 0.236 说明切分影响不大"不再作为依据——同一比较在另一口径下差 76 倍，说明至少一次口径错位，接近可能是巧合；
2. **查因不猜因**：用同一份预测文件按两种口径重算。比值在任意子集上恒定 ⇒ 纯尺度差（归一化），可安全统一；比值不恒定 ⇒ 分母/遮挡子集/通道口径差异，**测的东西不同**，必须修正后重跑；
3. **对内**：训练损失统一冻结 [0,1] 归一化（只读 train_stats），合法；
4. **对外（论文表格）**：一律来自 C 的冻结 evaluator，反归一化到物理单位（m/s 等）主报告——审稿可解释、免疫归一化漂移；
5. **永不**把本地数字与 KDD Table 3 数字放进同一张表；
6. **每次跑必留口径 manifest**（损失定义、遮挡协议、切分版本、归一化 stats 哈希）。1 epoch 0.003085 vs 17 epoch best 0.005358 的反向差异即口径漂移现行案例——禁止跨口径串联结论；
7. 早停规则（patience=10 + max epoch）写进冻结的 parent 校准预算，候选沿用同一规则。

## 五、通用 Research Kernel 的处置（回答 8/26 提问）

`SimpleAutoResearch` 中提炼的领域无关内核（`research/controller.py`、`decision.py`、`candidates.py`、`budget.py`、`stopping.py`、`campaign.py` 等 13 个模块 + `test_kernel_boundaries.py` 边界测试）**不随框架一起作废**，处置为"**运行时退役、资产保留**"：

- 作为 ICLR 运行时：退役——一个 discovery 只能有一个运行时（CLH），双运行时是人时灾难；
- 作为资产：保留——DecisionPolicy/预算/停止条件的**协议内容**已以 JSON 合同形式（00_contract）注入 CLH：思想已迁移，代码留档；
- 例外通道：若集成中发现 CLH 的轻量 loop 缺关键能力（候选生命周期、预算账本、断点重入），**按模块移植** kernel 对应文件到 CLH，不复活整个框架；
- 后续价值：kernel + 边界测试是平台线（SZAR / ICML 版）的起点，投稿后再议。

## 六、验收口径（何时算"可以开始正式 discovery"）

对照 campaign_state 的 `04_discovery` 解锁条件，CLH 侧新增三项：

- [x] trial 执行已子进程隔离，evaluator 进程不加载 agent 代码（P0 已修复并有测试，见 §二）；
- [ ] 一次端到端演练：真实 AeroWF 训练 1 个 trial（小预算），checkpoint/predictions/digest 全链路落盘，C 的 evaluator 出全网格指标；
- [ ] 随机臂演练 2 个 trial，产物与 LLM 臂 schema 一致。
