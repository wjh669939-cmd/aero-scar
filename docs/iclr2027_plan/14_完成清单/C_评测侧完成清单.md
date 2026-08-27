# C（评测侧 + Test Custodian）完成清单

> 状态基准：2026-08-26 深夜。**C1/C2 已冻结交付（提前于 8/29）**。

## 已完成 ✅

| 项 | 内容 | 对照 |
|---|---|---|
| C2 封存 v1→v3 | EVALUATION.7z 签收（哈希与 B 一致）；事件切片按 v2 重做；v3 补正事件规则文档哈希（2ea71b0c…A 侧验证一致） | 07 C2、12 §四·五 |
| 事件切片定义 v2 | HAZARD 类 ∨ 阵风（PRECIP 不单独入事件）、并块 ≤6 步（90min）不变、与 v1 差异写明、训练侧可复刻 | 12 §四·五 |
| C1 evaluator v1.0 冻结 | 独立包 + 子进程 CLI（--predictions/--trial-meta/--split/--out-dir）；不 import CLH；封存路径仅私有配置解析 | 12 §一 |
| endpoint 网格 | 预测 36 项（3 horizon × wind_x/y × MAE/RMSE × 3 机场）+ 分类（Macro-F1、三类 CSI、CSI_macro）+ 插补 24 项（固定 mask 四场景） | contract v2 + 07 C1 |
| 三态语义 + 异常计数 | invalid/failed=not_evaluated；NaN/Inf/shape/ID/越界九类计数 | 07 C1 第 4 条 |
| evaluation_manifest | 每次评测落盘代码/配置/数据口径/输入摘要 | 12 §三-2 |
| 退化剔除预注册 | ZSPD/ZSSS val 分类全 GOOD → 预注册剔除写入冻结 config；真实 val 验证过 | 12 §四-4 |
| 边界测试 | NaN/Inf/错 shape/ID 乱序缺失重复/越界/空提交全过 | 07 C1 |
| golden 向量 | 完整 val sample_id + 期望 metrics；fail-safe 行为 A 侧实测正确 | 12 §五-1 |
| bootstrap 块单位 | UTC 日期块（已冻结二选一） | 07 C1 第 6 条 |

## 剩余

| 项 | 内容 | 依赖 | 时点 |
|---|---|---|---|
| ⬜ 私有配置部署 | val 真值 + 私有配置装到 discovery 机器（D 的 3090）的 evaluator 专属目录；给 A 追加禁访 token | 运行时落定 | G-9 |
| ⬜ 真实联调 | D 的真 checkpoint predictions → 子进程 → 全网格落盘（唯一没走过的链路） | G-7/G-9 | G-10 |
| ⬜ `test_lock_state.json` 落盘 | 按 07 交付锁状态文件到 00_contract | — | 8/28 前 |
| ⬜ C3 接受规则冻结 | 与 A/导师用 D1 三 seed 方差报告标定 decision_policy 数值 | G-11 | 8/31 前 |
| ⬜ 插补 golden 补充说明 | `pred` 形状 [sample,runway_slot,scenario,minute,channel] 与 scenario 固定顺序写进 golden README（A 反馈：最易做错的接口点） | — | 顺手 |
| ⬜ C5 一次性认证 | 9/11–9/12，候选冻结后执行一次，即刻重锁 | 候选冻结 | 9/11 |

## 红线提醒

- 封存后退出候选讨论；discovery 期间只回聚合指标；
- 认证只执行一次，任何"再测一次"须书面留痕拒绝；
- **注意**：val 分类信号仅来自 ZBAA（ZSPD/ZSSS 退化）——若有人以分类指标提候选，提醒其统计效力受限（support：hazard 94 / precip 240）。
