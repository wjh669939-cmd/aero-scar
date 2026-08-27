# aerowf-v1 / 00_contract 产物状态表

> 创建：2026-08-25（A 侧前置工作）
> 规则：状态为 FROZEN 的文件修改必须升版本号 + digest + 书面理由；DRAFT 文件按各自依赖落定后冻结。

| 文件 | 状态 | 冻结条件 | 阻塞方 |
|---|---|---|---|
| `campaign_state.json` | active（随阶段推进更新） | — | — |
| `seeds.json` | **FROZEN** | 已冻结 | 无 |
| `action_registry_v1_draft.json` | DRAFT | 8/27 DEC-001（M 轴）+ 9/2 DEC-002（tier-2）落定后升 v1.0，8/31 冻结 | 导师决策 |
| `axis_lock_v1_draft.json` | DRAFT | AeroWF 修复版仓库布局落定后填实路径 | D（8/27 前） |
| `decision_policy_v1_draft.json` | DRAFT（结构定死，数值 TBD） | D1 方差报告（8/26）后填数值，8/31 与 C、导师共同冻结 | D1 报告 |
| `schemas/trial.schema.json` | **FROZEN** | 已冻结（新增字段向后兼容可加，必填项不动） | 无 |
| `schemas/result.schema.json` | **FROZEN** | 同上 | 无 |
| `data_digest` / `split_manifest` | **已登记（8/26）** | MODEL_TRAINING.7z=b50b0a92…；PACKAGE_SHA256SUMS=17e1b2bf…；materialization_manifest=7744acf8…（见 `data_digest_partial.txt`）；194/194 文件哈希通过 | 无 |
| `evaluator_version` | **C1 v1.0 已冻结交付（8/26 深夜，提前于 8/29）** | 独立包+子进程 CLI 符合 12 文档合同；边界测试全过；golden 在无私有配置机器上正确 fail-safe；36 预测 endpoint + 分类 + 24 插补 endpoint；退化剔除已预注册（ZSPD/ZSSS 分类全 GOOD，信号仅 ZBAA） | 无（联调剩真实 checkpoint 一轮） |
| `test_lock_state.json` | **C2 已封存（8/26）** | C 按 07 落盘 lock 文件即完成；封存哈希链已核验（EVALUATION.7z=220228aa…）；本机 sealed 副本已清除 | C 落盘 |
| `DOWNSTREAM_TASK_CONTRACT_v2.{json,md}` | **已入库（8/26 晚，ACTIVE）** | json sha256=2e55c519…；正式预测协议回归论文口径：T+1/4/8=15/60/120min × wind_x/wind_y；v1 分钟级七变量改名 H+1m/4m/8m 仅作扩展 | 无 |
| `DOWNSTREAM_TASK_CONTRACT_v1.json` | 已入库（**SUPERSEDED by v2**，留档审计） | sha256=b28d01a3… | 无 |
| `C2_认证封存_SHA256_v2.txt` | 已入库（8/26 晚，被 v3 取代） | 曾缺事件规则文档哈希 | 无 |
| `C2_认证封存_v3`（含 `危险事件切片定义_v2.md`） | **已受理（8/26 深夜）** | 补正完成：事件规则文档哈希已入清单（2ea71b0c…本地验证一致）；v2 事件定义=HAZARD 类∨阵风（PRECIP 不单独入事件），并块 ≤6 步不变；切片 npz 哈希与 v2 清单一致 | 无 |
| `axis_lock_v1_draft.json` | **v1.0-RC（8/26 深夜）** | D 的 CODE_MAP 已把 4 个文件级问题全部回答，路径已填实；剩一项 G1 改造前提：两个下游脚本混含特征/损失/训练循环，须抽薄成独立模块后文件锁才成立 | A+D（随 G1） |

## 待决事项（同 campaign_state.pending_decisions）

- **DEC-001**（8/27）：受约束 Model 轴是否纳入 → 影响 registry 中 M1–M3 与 axis_lock 中 model 轴的 active 状态；
- **DEC-002**（9/2）：tier-2 是否启用 → 影响 O4–O6；
- **DEC-003**（8/27）：预训练语料含 ZBAD 的处置 → 影响空间认证主张措辞。

三个决策的 default（未决时）均已写入 campaign_state，不阻塞冻结日。
