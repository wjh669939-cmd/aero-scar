# C 评测侧：evaluator v1.0 对接 CLH 的修改要求 v1.1

> 日期：2026-08-25 首版，2026-08-26 晚按 DOWNSTREAM_TASK_CONTRACT **v2** 修订（发出前更新，C 收到的即本版）
> 收件人：C　|　背景：CLH 已定为 discovery 运行时（10/11 文档）；本文件把 07 任务书 C1 的要求**具体化为对接合同**，是 07 的增补而非替代，冻结日仍为 8/29。
> **口径基准**：任务层语义、指标定义、zero-support 规则一律以 `DOWNSTREAM_TASK_CONTRACT_v2`（B 8/26 发布）为准；本文件只补充"接口形态 + CLH 对接 + 治理性检查"，与 contract 冲突时以 contract 为准并知会 A。

---

## 一、交付形态（最重要的变化）

1. **独立包 + 子进程调用，禁止被 harness import**。CLH 现版把 agent 代码加载进 evaluator 进程（10 文档 P0 漏洞），修复后架构为：agent 训练产物 → predictions 文件 → 你的 evaluator 子进程。你的代码不 import CLH，CLH 不 import 你的代码，唯一接口是文件。
2. 命令行入口（示例，参数名可议但 8/29 后冻结）：

```bash
python -m aerowf_evaluator \
  --predictions /path/predictions.npz \
  --trial-meta /path/trial_meta.json \
  --split val \
  --out-dir /path/out/
```

1. **封存数据路径只在你的私有配置里解析**，不接受调用方传入（防止路径出现在 agent 可见日志中）；`--split` 只接受 `val`；认证走独立入口 `certify`，仅你本人可执行。



## 二、输入合同

- `predictions.npz`：`pred`（float 数组）+ `sample_id`（与 split manifest 对齐）；读取一律 `allow_pickle=False`；
- `trial_meta.json`：trial_id、arm、seed、task、checkpoint digest（只用于登记，不影响评分）。



## 三、输出合同

1. `metrics.json`：全 endpoint 网格 + 三态状态（completed/invalid/failed）+ 异常计数（NaN/越界/缺失 ID）。**只含聚合值，无逐样本分数**；
2. `evaluation_manifest.json`：evaluator 版本 digest、split manifest 哈希、归一化 stats 文件哈希、指标定义标识、样本数、UTC 时间——**每次评测都写，这是口径漂移的唯一防线**（10 文档 §四第 6 条的执行机制）。



## 四、指标口径（按 contract v2 修订）

1. **endpoint 网格以 contract v2 为准**：
  - 正式预测：**{T+1=15min, T+4=60min, T+8=120min} × {wind_x, wind_y} × {MAE, RMSE} × 3 机场**（跑道级、`runway_mask=True` 有效点，时间戳对齐、伙伴窗内部索引 95）；
  - 分类：Macro-F1、CSI_GOOD/PRECIP/HAZARD/CSI_macro × 3 机场，21→3 映射、ignore_index=-100、zero-support 政策全按 contract v2 §3–§5；
  - v1 的分钟级七变量网格**不进正式表**；若保留为扩展实验，endpoint 名一律用 `H+1m/H+4m/H+8m` 前缀，与正式 T+1/T+4/T+8 物理隔开，禁止混排；
  - 07 任务书里的 "Crosswind MAE" 等派生 endpoint 如保留，须在 config 中声明由 wind_x/wind_y 后处理派生，且只作次要 endpoint；
2. **主指标尺度 = 释放的归一化 [0,1]**（contract v2 §4 定案，13 文档 §四的争议就此关闭）：`*_source_scale` 只作次要诊断，本文件 v1.0 的"物理单位主报告"条款作废；口径漂移防线改由 evaluation_manifest（本文件 §三-2）+ contract_sha256 随行承担；
3. **废弃 CLH 占位口径**：τ=0.005、等权 Ī、`overall.mae` 单值——你的 evaluator 只测量（输出网格），**不判定接受**；接受判定由 DecisionPolicy（A 侧，数值待 D1 方差报告）消费你的 metrics.json 完成。测量与判定分离；
4. **endpoint 退化检测（源于 8/25 试跑）**：试跑中 hazard CSI=1.0 是退化值。你需要在真实 val 上对每个 endpoint 做 sanity：报告各机场分类三类支持度与事件基率；基率为 0 或 1 的 endpoint 标记 `degenerate` 并按预注册规则从网格剔除（剔除规则写进 evaluator config 一并冻结，不得事后临时剔除）；contract v2 的 zero-support（NA + 分母剔除 + FP 单列）覆盖分类侧，本条覆盖其余 endpoint；
5. **命名纪律**：`class_hazard`（分类 HAZARD 类，不含降水）与 `event_slice`（认证事件切片）是两个概念，metrics.json 字段名与 config 中不得混用裸词 "hazard"；
6. 其余沿用 07 任务书 C1：固定 mask、零分母报 null、越界禁 clip、block-bootstrap 块单位（日期块或事件块，冻结前二选一）。



## 四·五、C2 v2 封存的一项补正请求

C2 二次封存（v2 事件切片）已受理：EVALUATION.7z 哈希不变（220228aa…），四个私有事件切片 npz 已换 v2 哈希。**但 v2 封存清单缺少"v2 事件切片规则文档"本身的哈希**（v1 清单里有 `危险事件切片定义.md` = f1697db9…，v2 清单没有对应条目）。请补两件事：

1. 出一版 `危险事件切片定义_v2.md`（写明 v2 切片依据：是沿用 v1 规则"label∈[3,20] ∨ has_gust，间隔≤6 步并块"，还是改用 contract v2 的 HAZARD 类定义[不含降水]，以及并块参数是否变化），训练侧复刻要用；
2. 把该文档哈希追加进 C2 封存清单（升 v3 或追加行均可）——没有规则版本哈希，"切片版本已固定"的主张不完整。



## 五、联调安排

1. **8/27 前**请提供一套 **golden 测试向量**：小型伪造 predictions.npz + 对应的期望 metrics.json——A 用它做 CLH 适配器的 mock 联调，你冻结前的接口变更只需重发 golden 文件；
2. 8/28–8/29 真实联调：D 的一个小预算 checkpoint 产出 predictions → 你的子进程 → 全网格指标落盘；
3. 冻结（8/29）后接口不再变，新增 endpoint 走版本升级流程。



## 六、验收清单（对接部分，补充 07 的 C1 验收）

- [ ] harness 进程零 import evaluator 代码（子进程边界确认）；
- [ ] golden 向量联调通过（golden 按 contract **v2** 网格生成：3 horizon × wind_x/wind_y）；
- [ ] metrics.json 记录 `task_contract_version="2.0"` 与 contract_sha256；normalized 主口径，`*_source_scale` 如输出仅作次要；
- [ ] 退化 endpoint 检测在真实 val 上跑过一遍，剔除清单（如有）已预注册；
- [ ] 每次评测产出 evaluation_manifest.json，抽查字段齐全；
- [ ] `危险事件切片定义_v2.md` 已出且哈希已入 C2 封存清单（§四·五）。

---



## 变更记录


| 版本   | 日期      | 变更                                                                                                                                                                  |
| ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1.0 | 8/25 深夜 | 初版（未发出）                                                                                                                                                             |
| v1.1 | 8/26 晚  | 发出前按 DOWNSTREAM_TASK_CONTRACT v2 修订：endpoint 网格改 3 horizon（15/60/120min）× wind_x/wind_y；主口径定为归一化 [0,1]（v1.0"物理单位主报告"条款作废）；新增 §四·五 C2 v2 封存补正请求（事件规则文档哈希缺失）；新增命名纪律条款 |


