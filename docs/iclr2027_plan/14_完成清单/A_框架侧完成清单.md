# A（框架/harness 侧）完成清单

> 状态基准：2026-08-26 深夜

## 已完成 ✅

| 项 | 内容 | 证据位置 |
|---|---|---|
| 契约脚手架 | campaign_state、seeds（v1.1）、trial/result schema（冻结）、decision_policy 草案、action registry 草案（15 动作 + 9 个 param_space） | `workspace/runs/aerowf-v1/00_contract/` |
| A-1 schema 校验器 | trial/result 合规 + M 轴参数预算 gate | `tools/contract_tools/validate.py` |
| A-2 axis-lock 引擎 | 越界/非活跃轴/空 diff/隐藏路径 token 全拒绝 | `tools/contract_tools/axis_lock.py` |
| A-3 随机臂采样器 | seed 可复现、active 模板、param_space 采样 | `tools/contract_tools/random_arm.py` |
| A-4 提案解析器 + 提示词 v0.1 | 严格 JSON、科学字段无默认、自由提案标记 | `tools/contract_tools/proposal_parser.py` |
| A-5 CLH P0 修复 | 子进程隔离 + eval 标签剥离（封堵回显作弊）+ 白名单环境；19 项测试全绿 | CLH `src/clh/research/{subproc,trial_runner,evaluator}.py`、`tests/test_subproc.py` |
| A-6 evaluator 客户端 | 对齐 12 文档 CLI，三态语义；golden fail-safe 已实测 | `tools/contract_tools/evaluator_client.py` |
| G3 轴配置对齐 | CLH 增 objective 轴，atc.toml = R+O 双轴 | CLH `configs/atc.toml` |
| CLH 审计 + 运行时决策 | radar harness 退役、CLH 唯一运行时、kernel 资产保留 | 10/11 文档 |
| 交付受理 ×3 轮 | B 双包/contract v1→v2/C1+C2 v3 全部核验登记 | 13 文档、00_contract/README |
| axis_lock 路径填实 | CODE_MAP 四问全答，升 v1.0-RC | `00_contract/axis_lock_v1_draft.json` |
| 监管链清理 | 本机 EVALUATION 封存副本删除留痕 | `local/data_side/CUSTODY_REMOVAL_LOG.txt` |
| 单测 | contract_tools 28 项全绿 | `tools/tests/` |

## 8/27 新完成 ✅

| 项 | 内容 | 证据位置 |
|---|---|---|
| G-7 执行器（本地版） | `pipeline_executor.py`：HANDOFF §4 调用模板 + §6 六条件判定 + 八条失败线（NaN/Inf 扫描、目录冲突拒绝、超时、现场保留、checkpoint SHA）；假 pipeline 11 项单测全绿 | `tools/contract_tools/pipeline_executor.py` + `tests/test_pipeline_executor.py` |
| G-8 抽薄方案 | trial_features / trial_objective 拆分规格（抽什么、不抽什么、接口签名、move-only 纪律、seed43 一致性验收）交 D 执行 | `15_G8抽薄改造方案.md` |
| axis_lock 修复升级 | JSON 恢复机器可解析 allow（双根占位符 `<AEROWF_REPO>`/`<DOWNSTREAM>`，抽薄后真实路径）；引擎改绝对路径匹配；测试改真实路径并新增训练循环拒绝/ZBAD token 用例 | `00_contract/axis_lock_v1_draft.json`、`contract_tools/axis_lock.py` |
| G-5 提示词合同 + 组装器 | 模板升 v0.2 嵌入四轴 INTERFACE CONTRACT；`context_assembler.py`：registry active 模板 + lineage 摘要 + failure slices 组装，组装后强制隐藏 token 拦截（命中即停机） | `contract_tools/context_assembler.py` + prompts v0.2 |
| G-9 部署清单 | 传输物/独立 venv/验收七步/红线（EVALUATION 不进 D 机、GPU 分时建议） | `16_部署清单_D机3090.md` |
| 单测 | 48 项全绿（28→48） | `tools/tests/` |

## 8/27 凌晨部署完成 ✅

| 项 | 内容 | 结果 |
|---|---|---|
| 部署执行 | 4.3M 部署包（CLH+tools+合同区+计划文档+交接文档+C1 评测器+两篇论文 PDF）scp 到 D 机，SHA-256 两端一致；数据符号链接挂载 release_v1 | `/root/autodl-tmp/clh_deploy/` |
| D 机测试 | contract_tools 48 项 + CLH 23 项（含真实数据 4 项，0 跳过）全绿 | — |
| 部署中发现并修复的回归 | ① `offline.py` 轴识别正则/preset 映射漏 objective 轴（G3 改配置时遗漏，本地无数据测不出）；② `test_aerowf.py` 断言仍按旧 4 轴配置写 | 修复已同步本机与 D 机 |
| LLM API | `.env`（CLH_API_KEY，600 权限）配到 CLH 目录；DeepSeek 烟测通过（deepseek-v4-flash） | — |
| G-7 真机验收 | 执行器驱动真实 full_pipeline（seed 1001，1 epoch）：**success，六条件全过，693.8s** | `results/harness/trial_20260827_003243_seed1001_c71c23` |
| evaluator fail-safe | D 机上无 C 私有配置时正确拒评（status=failed, not_evaluated） | 待 C 部署私有配置转绿 |

## 剩余（依赖外部）

| 项 | 内容 | 依赖 | 预计 |
|---|---|---|---|
| ✅ G-10 收尾 | 8/27 双腿 completed，全网格落盘，两缺口关闭 | — | 已闭 |
| ✅ decision_policy 数值标定 | v1.1 已标定并随 8/27 决议冻结（imputation 定义提案待 C 加映射键） | — | 已闭 |
| ✅ 随机臂演练 | G-13 两趟全链路通过（rand-rep-000 / rand-obj-000） | — | 已闭 |
| ✅ LLM 臂冒烟→正式开跑 | discovery 8/27 16:23 启动；batch01 两败（代码 bug）→ 补 G-15 冒烟闸门；batch02 在跑 | — | 已闭 |
| 🔶 迭代闭环补齐 | evidence_builder（机器证据表）✅、裁决规则 v1 ✅、父子链 ✅；自由提案方案（22 文档）待拍板落地 | trial 3 结果 | 8/28 前 |
