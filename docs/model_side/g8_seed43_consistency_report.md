# G-8 抽薄一致性详细报告（seed 43）

- 口径：同一冻结数据、同一 seed=43、同一 Contract V2、同一训练超参；仅下游脚本改为 trial_features / trial_objective 抽薄布局。
- 原目录：`results/full_pipeline/seed43_v2`
- 复跑目录：`results/full_pipeline/seed43_v2_g8_consistency`（未覆盖原结果）
- 判定：**非易失指标 bit 级一致**（max |Δ| = 0）。

## 1. 验收结论

抽薄是 move-only：特征构造与损失函数换了文件位置，没有改变计算图或随机流。五阶段 `metrics.json` 在去掉耗时/SHA/环境等易失字段后，数值与结构完全相同。

| 项 | 原 seed43 | G-8 复跑 |
|----|-----------|----------|
| pipeline status | success | success |
| 墙钟（秒） | 9823.6 | 9542.1 |
| 非易失 pipeline_summary 差分条数 | — | 0 |

墙钟不同是机器负载差异，不进入一致性判定。

## 2. 代码布局

复跑时磁盘上的下游脚本已是抽薄版（与 handoff 里未抽薄 SHA 不同），并 import：

- `src/trial_features.py`（R 轴）
- `src/trial_objective.py`（O-tier1）

- live forecast SHA-256: `39c6490e4a20412cbc4f51f48acf052161e9958d53093c524171780c92bb0547`
- original forecast SHA-256: `f4031081b20b81f5cf1578b3cb814bf99bac5cffd8f454ff9ebbfa664f1ee4a4`
- live classification SHA-256: `974d77c01f54d5f84d0e99b0a7f82be2342d805947661cae3f85031749a21140`
- original classification SHA-256: `91c68814cd829ae8908c55663f452cc209563d4818eec0ebe009bfdf0321c8ad`

预训练脚本本轮未抽，两边 `source_sha256.pretrain` 仍为同一哈希。

## 3. 五阶段主指标

### 预训练

| 指标 | 原 | G-8 | Δ |
|------|----|-----|---|
| val_loss | 0.005464384305325 | 0.005464384305325 | 0.0 |
| train_loss | 0.004193576505968 | 0.004193576505968 | 0.0 |
| 配置 epochs | 100 | 100 | 0 |
| 墙钟（秒） | 4736.4 | 4586.1 | 易失 |

### Forecast Scratch

| 指标 | 原 | G-8 | Δ |
|------|----|-----|---|
| epochs_completed | 28 | 28 | 0 |
| best_epoch | 18 | 18 | 0 |
| best_val_mse_norm | 0.002323424018879 | 0.002323424018879 | 0.0 |

| Horizon | 原 RMSE | G-8 RMSE | 原 MAE | G-8 MAE |
|---------|---------|----------|--------|---------|
| T+1 | 0.044362520239 | 0.044362520239 | 0.022001087781 | 0.022001087781 |
| T+4 | 0.048851265762 | 0.048851265762 | 0.025141639593 | 0.025141639593 |
| T+8 | 0.051144820126 | 0.051144820126 | 0.027892693368 | 0.027892693368 |

### Forecast Pretrained

| 指标 | 原 | G-8 | Δ |
|------|----|-----|---|
| epochs_completed | 30 | 30 | 0 |
| best_epoch | 30 | 30 | 0 |
| best_val_mse_norm | 0.002741874563842 | 0.002741874563842 | 0.0 |

| Horizon | 原 RMSE | G-8 RMSE | 原 MAE | G-8 MAE |
|---------|---------|----------|--------|---------|
| T+1 | 0.051133734293 | 0.051133734293 | 0.028521174493 | 0.028521174493 |
| T+4 | 0.052261060753 | 0.052261060753 | 0.028660601024 | 0.028660601024 |
| T+8 | 0.053663268877 | 0.053663268877 | 0.030580309919 | 0.030580309919 |

### Classification Scratch

| 指标 | 原 | G-8 | Δ |
|------|----|-----|---|
| epochs_completed | 21 | 21 | 0 |
| best_epoch | 11 | 11 | 0 |
| macro_f1 | 0.727509914954604 | 0.727509914954604 | 0.0 |
| CSI_macro | 0.610294143593331 | 0.610294143593331 | 0.0 |
| accuracy | 0.976436926702011 | 0.976436926702011 | 0.0 |

| 类 | 原 F1 | G-8 F1 | 原 CSI | G-8 CSI |
|----|-------|--------|--------|---------|
| GOOD | 0.987839833232 | 0.987839833232 | 0.975971852742 | 0.975971852742 |
| PRECIP | 0.644128113879 | 0.644128113879 | 0.475065616798 | 0.475065616798 |
| HAZARD | 0.550561797753 | 0.550561797753 | 0.379844961240 | 0.379844961240 |

### Classification Pretrained

| 指标 | 原 | G-8 | Δ |
|------|----|-----|---|
| epochs_completed | 30 | 30 | 0 |
| best_epoch | 23 | 23 | 0 |
| macro_f1 | 0.734911257633807 | 0.734911257633807 | 0.0 |
| CSI_macro | 0.620618756355748 | 0.620618756355748 | 0.0 |
| accuracy | 0.979887233863503 | 0.979887233863503 | 0.0 |

| 类 | 原 F1 | G-8 F1 | 原 CSI | G-8 CSI |
|----|-------|--------|--------|---------|
| GOOD | 0.989669332181 | 0.989669332181 | 0.979549927270 | 0.979549927270 |
| PRECIP | 0.688346883469 | 0.688346883469 | 0.524793388430 | 0.524793388430 |
| HAZARD | 0.526717557252 | 0.526717557252 | 0.357512953368 | 0.357512953368 |

## 4. 非易失字段穷尽比对

对每个阶段的 `metrics.json` 去掉耗时、GPU 峰值、环境、各类 sha256 后递归比较。

| 阶段 | 非易失差分条数 | 数值 max\|Δ\| | 说明 |
|------|----------------|---------------|------|
| pretrain | 0 | 0 | 完全一致 |
| forecast_scratch | 0 | 0 | 完全一致 |
| forecast_pretrained | 1 | 0 | 仅 `checkpoint_path` 目录名不同（`seed43_v2` vs `seed43_v2_g8_consistency`） |
| classification_scratch | 0 | 0 | 完全一致 |
| classification_pretrained | 1 | 0 | 同上，仅预训练权重路径字符串 |

没有任何浮点数值差异。`pipeline_summary.json` 在去掉 `paths` / 耗时 / sha256 后差分条数为 0。两边记录的 **脚本 SHA 不同**（抽薄后 forecast/classification 源文件已变），这是预期，不计入指标一致性。

## 5. 产物 SHA-256（含易失内容）

`metrics.json` 哈希不同是因为里面写了墙钟、环境、checkpoint 绝对路径；**验证集预测 `validation_predictions.npz` 四个下游任务哈希全部相同**，说明逐样本输出 bit 级一致。`best_model.pth` 哈希不同常见于序列化元数据，不单独作为失败条件。

| 阶段 | 文件 | 哈希是否相同 |
|------|------|--------------|
| pretrain | `metrics.json` | 否 |
| pretrain | `checkpoints/best_model.pth` | 否 |
| forecast_scratch | `metrics.json` | 否 |
| forecast_scratch | `history.csv` | 否 |
| forecast_scratch | `checkpoints/best_model.pth` | 否 |
| forecast_scratch | `validation_predictions.npz` | 是 |
| forecast_pretrained | `metrics.json` | 否 |
| forecast_pretrained | `history.csv` | 否 |
| forecast_pretrained | `checkpoints/best_model.pth` | 否 |
| forecast_pretrained | `validation_predictions.npz` | 是 |
| classification_scratch | `metrics.json` | 否 |
| classification_scratch | `history.csv` | 否 |
| classification_scratch | `checkpoints/best_model.pth` | 否 |
| classification_scratch | `validation_predictions.npz` | 是 |
| classification_pretrained | `metrics.json` | 否 |
| classification_pretrained | `history.csv` | 否 |
| classification_pretrained | `checkpoints/best_model.pth` | 否 |
| classification_pretrained | `validation_predictions.npz` | 是 |

## 6. 方案对照

| G-8 步骤 | 结果 |
|----------|------|
| move-only 抽薄 | 已落地 `trial_features.py` / `trial_objective.py` |
| smoke seed 1001（1+1 epoch） | success，约 410s |
| seed43 复跑到新目录 | `seed43_v2_g8_consistency`，FULL PIPELINE SUCCESS |
| metrics 一致性 | bit 级一致，Δ=0 |
| axis_lock 升 FROZEN | 待 A；D 可通知 |
| 两笔 git commit | 待明确要求后再做 |

## 7. 结论与后续

文件级 axis-lock 的工程前提已满足：R/O 对象已从锁定主脚本抽出，主脚本只保留训练循环、目标生成、`map_labels` 与泄漏断言。抽薄未引入行为变化。

建议：把 `aerowf_forecast_train_v2.py` 与 `aerowf_classification_train_v2.py` 列入 `locked_paths_always`，通知 A 将 axis_lock 从 RC 升为 FROZEN。

