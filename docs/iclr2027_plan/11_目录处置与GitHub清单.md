# 目录处置与 GitHub 发布清单 v1.0

> 日期：2026-08-25　|　磁盘现状：150G 总量，已用 21%，**无空间压力**——大件删除不紧迫，风险收益比决定"标记优先于删除"。
> 处置标签：**GITHUB**（随论文发布，匿名化后上传）/ **KEEP**（本地保留，审计凭证）/ **RETIRED**（保留但冻结，禁止改动）/ **DEL-NOW**（已删）/ **DEL-AFTER-SUBMIT**（投稿后可删候选，现在不删）

---

## 一、运行时决策（定论）

**radar harness（SimpleAutoResearch）退役，CLH 为唯一 discovery 运行时；协议以 JSON 合同形式注入 CLH，不做反向融合。**

理由：旧框架为雷达图像任务（[B,T,C,H,W]、PNG 管线、SEVIR 接口）深度定制，改造到 AeroWF 表格时序的成本高于 CLH 补齐 4–6 人天缺口；CLH 已用真实 API 验证协议骨架；两套并行维护是人时灾难。已在退役仓库根写入 `RETIRED_NOTICE.md`。

## 二、处置清单

| 路径 | 大小 | 处置 | 说明 |
|---|---|---|---|
| `closed loop research/` | <1M | **GITHUB** | 论文主代码库。上传前：匿名化、删 `runs/` 试跑产物与 `.env.example` 中网关信息、补 LICENSE |
| `workspace/runs/aerowf-v1/` | 40K | **GITHUB** | 00_contract 全部 JSON 合同 + 未来 trial 谱系/认证记录 = 论文可复现性材料主体 |
| `local/iclr2027_plan/` | 小 | **KEEP** | 内部计划文档，不上传（含时间线与人员信息） |
| `workspace/repos/AutoResearch/SimpleAutoResearch/` | ~128M | **RETIRED** | 已标记退役；G1 案例若进论文附录，引用协议文档而非代码 |
| `workspace/runs/radar-paper-v1/` 中的 json/log/md 记录 | ~1M | **KEEP** | G1 拒绝案例的审计凭证（g1_acceptance、gate_status、diagnostic、resource_usage 等），论文方法节要用 |
| `workspace/runs/radar-paper-v1/01_parent_calibration` 逐 epoch checkpoint + npz | ~14G | **DEL-AFTER-SUBMIT** | 被拒 campaign 的中间产物；digest manifest（小文件）保留，二进制投稿后删 |
| `workspace/runs/radar-paper-v1/02_strong_parent` 大件 | ~3.4G | **DEL-AFTER-SUBMIT** | 同上；若删，建议保留 3 seed 的 E*=25 checkpoint 各一份 |
| `workspace/runs/{radar-pilot-*, llm-radar-actionspace-v1}` 大件 | ~8.4G | **DEL-AFTER-SUBMIT** | 内部 pilot，json 记录保留，checkpoint/预测可删 |
| `workspace/runs/sevir-v1/` | 5.3M | **KEEP** | 已搁置 campaign 的合同与转移记录，小，留作决策链凭证 |
| `workspace/data/radar_*`（tar + raw + v1_full） | ~5.9G | **DEL-AFTER-SUBMIT** | 雷达数据整体退役；tar 与 raw 疑似同源冗余，如需先删可只删 `radar_png_full.tar`（433M） |
| `workspace/repos/ZhiXiang*`、`workspace/incoming/eval_v1.3_final` | ~8M | **RETIRED** | 雷达/SEVIR 时代的模型与评测包，冻结保留 |
| 全部 `__pycache__` / `*.pyc` | — | **DEL-NOW（已执行）** | 已删 64 个缓存目录 |
| AeroWF 修复版训练仓库（D 机器） | — | **GITHUB** | AeroWF† 修复清单 + 调参日志随论文公开（03 文档预答复的依据），提醒 D 纳入版本管理 |
| C 的 evaluator 包（C 机器/独立目录） | — | **GITHUB（代码）；封存数据永不上传** | evaluator 代码公开支撑可信度；sealed 数据与 access log 不出本地 |

## 三、GitHub 发布前检查单（投稿周执行，先登记在此）

- [ ] CLH：删除 API 网关地址、试跑 runs、内部路径（`../AeroWF/数据接口/...` 改相对配置）；
- [ ] 匿名化：所有文档去机构/人名/机场私有标识核查（Aero-Dataset 本身已随 KDD 公开）；
- [ ] `aerowf-v1` 谱系：认证前后的完整 trial 账本 + digest 清单导出；
- [ ] evaluator：代码 + config + 边界测试用例上传；sealed manifest 只上传哈希；
- [ ] 复现脚本：从冻结 commit + manifest 一键重建主表数字的入口脚本。
