# D 侧 discovery 值守说明（8/28 起生效）

> 目标：D 在白天档独立监护 discovery 批次，A 远程兜底。全部操作在 D 机本机。

## 一、目录与关键文件

| 路径 | 用途 |
|---|---|
| `/root/autodl-tmp/clh_deploy/discovery/discovery_runner.py` | 驱动（勿改；改动权在 A，走 git 归档） |
| `/root/autodl-tmp/clh_deploy/discovery/batch*.log` | 批次日志（`TRIAL_OUTCOME` / `BATCH_DONE` 行是结论） |
| `/root/autodl-tmp/clh_deploy/discovery/lineage.jsonl` | 谱系（只追加，**严禁编辑/删除已有行**） |
| `/root/autodl-tmp/clh_deploy/discovery/trials/<id>/` | 每个 trial 的提案/编辑/评测/result.json |
| `/root/autodl-tmp/aerowf_downstream_v2/results/harness/` | 训练输出与 `<run_id>.stdout/.stderr`（失败现场，**保留勿删**） |

## 二、启动一批（示例：R 轴 + O tier-1 + O tier-2 各一发）

```bash
cd /root/autodl-tmp/clh_deploy/local/eval_side/C1+C2/C1_评测器_v1
source /root/autodl-tmp/c_evaluator_private/evaluator.env
nohup /root/miniconda3/bin/python -u /root/autodl-tmp/clh_deploy/discovery/discovery_runner.py \
  --api-key <向 A 索取> \
  --plan representation,objective_tier1,objective_tier2 \
  --seed 42 --start-seq <上一批最后序号+1> \
  > /root/autodl-tmp/clh_deploy/discovery/batch<NN>.log 2>&1 &
```

要点：
- **必须**在 C1 评测器目录下、source 过 `evaluator.env` 的 shell 里启动；
- `--start-seq` 递增不回退（trial_id 唯一性）；
- 每发约 2.8h；排批前算好结束时间，**给夜间档（多 seed 确认 / parent 补跑）留整块**；
- API key 不落盘、不进 git、不写进任何脚本（链式启动用 `chain_batch*.sh`，key 走进程参数）。

**8/28 起 objective_tier2 可排**（驱动已接线）：tier2 编辑走函数级拼接协议——LLM 只产出
`unified_pretrain_forward` 替换实现，函数段之外机器保证不变；每个 tier2 trial 的
`trials/<id>/tier2_function_diff.txt` 是函数 diff（人审兜底材料，出结果前扫一眼）。
另：自由提案若与模板机制实质等价会被"假自由拦截"记 `proposal_rejected`，属正常事件。

## 三、监护与判读

- 看进度：`tail -f batch<NN>.log`；GPU：`nvidia-smi`；
- 每个 trial 结束打印一行 `TRIAL_OUTCOME`，字段判读：
  - `status=completed` + `screen_pass=true` → **过筛**，立刻通知 A（进 3-seed 确认队列）；
  - `status=completed` + `screen_pass=false` → 正常负结果，无需动作；
  - `event=smoke_rejected / proposal_rejected / codegen_rejected` → 无 GPU 损耗，无需动作；
  - `status=failed` → 看 lineage 该行的 `stderr_tail`；**不重跑、不删现场**，报 A 定性；
- `TRIAL_DRIVER_ERROR` 或批次中断 → 报 A（附 log 尾部 30 行）。

## 四、红线（与任务书一致，重申）

1. 不修改 `trial_features.py` / `trial_objective.py` 之外任何被 trial 触碰的文件；发现驱动把轴文件还原失败（对照 `src/trial_*.py` 与 git 归档版），先停批次再报 A；
2. 不进入 `/root/autodl-tmp/c_evaluator_private/`（禁访 token 会拦，但请勿测试它）；
3. 失败 trial 的输出目录、stdout/stderr 一律保留；磁盘紧张时报 A 统一清理；
4. lineage.jsonl 只增不改；
5. 正式补跑（parent 5-seed、确认跑）与 discovery 批次**不并行**（单卡显存与吞吐都不够）。

## 五、今晚交接（8/27 → 8/28）

- batch02（trial 2/3）预计 **00:20 前后**结束，结论 A 会同步；
- 之后夜间档跑 **parent seed 3407 / 5519**（D3 收尾，约 0:30–5:40）；
- 8/28 白天档第一批由 A 发起（自由提案方案落地后），之后逐步移交 D 按本说明操作。

---

*A 侧起草 2026-08-27 晚；有不清楚的先问 A，宁可停也不要猜。*
