# AeroWF Model-side Baseline and Harness Handoff

## 1. 交付状态

AeroWF v2 模型侧验证集全流程已经跑通：

1. Unified pretraining
2. Forecast scratch
3. Forecast pretrained
4. Classification scratch
5. Classification pretrained
6. Validation candidate selection

Seed 43 全流程最终审计结果：

- 所有阶段 status=success
- 所有阶段 test_used=false
- 预训练权重依赖检查通过
- Classification 输入泄漏检查通过
- Checkpoint SHA-256 检查通过
- Sealed-test isolation 检查通过

当前验证候选：

- Forecast：scratch
- Classification：pretrained

该选择仅基于验证集，不是密封测试集最终结果。

## 2. 目录位置

代码仓库：

/root/autodl-tmp/aerowf_baseline/AeroWF

冻结模型训练数据：

/root/autodl-tmp/aerowf_delivery/v1/extracted/AeroWF_v1_MODEL_TRAINING/release_v1

下游任务工作区：

/root/autodl-tmp/aerowf_downstream_v2

任务协议：

/root/autodl-tmp/aerowf_downstream_v2/contracts/DOWNSTREAM_TASK_CONTRACT_v2.json

Seed 43全流程结果：

/root/autodl-tmp/aerowf_downstream_v2/results/full_pipeline/seed43_v2

## 3. 运行环境

详细配置见：

- environment/system.txt
- environment/nvidia_smi.txt
- environment/gpu_summary.csv
- environment/runtime.json
- environment/runtime_validation.txt
- environment/pip_freeze.txt
- environment/pip_check.txt

当前关键环境：

- GPU：NVIDIA GeForce RTX 3090
- GPU显存：24 GiB
- PyTorch：2.5.1+cu124
- PyTorch构建CUDA：12.4
- GPU驱动报告CUDA兼容上限：12.8
- Python：3.12.3
- 训练峰值显存约：3.3 GiB
- SoftDTW CUDA：已验证可运行
- UnifiedTrainer HAS_DTW：True

注意：

nvidia-smi中的CUDA版本表示驱动兼容上限，不等于PyTorch实际构建CUDA版本。复现时应以torch.version.cuda和runtime.json为准。

## 4. Harness统一接入入口

统一控制器：

/root/autodl-tmp/aerowf_downstream_v2/src/aerowf_full_pipeline_v2.py

Harness应从以下工作目录启动：

/root/autodl-tmp/aerowf_baseline/AeroWF

正式调用模板：

cd /root/autodl-tmp/aerowf_baseline/AeroWF

python -u \
  /root/autodl-tmp/aerowf_downstream_v2/src/aerowf_full_pipeline_v2.py \
  --seed <SEED> \
  --batch-size 128 \
  --pretrain-epochs 100 \
  --downstream-epochs 30 \
  --patience 10 \
  --min-delta 1e-4 \
  --num-workers 0 \
  --output-root <UNIQUE_OUTPUT_ROOT>

每个Harness trial必须使用独立输出目录，例如：

/root/autodl-tmp/aerowf_downstream_v2/results/harness/<RUN_ID>

不得复用其他trial的正式输出目录。

## 5. Harness快速Smoke Test

Harness接入完成后的最小验证命令：

cd /root/autodl-tmp/aerowf_baseline/AeroWF

python -u \
  /root/autodl-tmp/aerowf_downstream_v2/src/aerowf_full_pipeline_v2.py \
  --seed 1001 \
  --batch-size 128 \
  --pretrain-epochs 1 \
  --downstream-epochs 1 \
  --patience 10 \
  --min-delta 1e-4 \
  --num-workers 0 \
  --output-root \
  /root/autodl-tmp/aerowf_downstream_v2/results/harness/smoke_seed1001

Smoke Test只用于验证接口，不作为论文正式结果。

## 6. Harness任务接口

### 输入

Harness通过命令行参数控制：

- seed
- batch size
- pretraining epochs
- downstream epochs
- early-stopping patience
- minimum improvement delta
- DataLoader workers
- unique output root

各脚本完整参数见：

- pretrain_help.txt
- forecast_help.txt
- classification_help.txt
- full_pipeline_help.txt

### 输出

统一控制器将生成：

- pretrain/
- forecast_scratch/
- forecast_pretrained/
- classification_scratch/
- classification_pretrained/
- pipeline.log
- pipeline_summary.json
- pipeline_summary.csv

每个训练阶段通常包含：

- metrics.json
- config.json
- history.csv
- validation_predictions.npz
- checkpoints/best_model.pth
- run.log

### 成功判定

Harness不得只解析终端文字。

成功条件应同时满足：

1. 进程退出码为0
2. pipeline_summary.json存在
3. 五个阶段metrics.json中的status均为success
4. 五个阶段test_used均为false
5. 所有best_model.pth存在
6. Pretrained任务checkpoint_load中missing_keys和unexpected_keys均为空

### 失败判定

以下任一情况均视为trial失败：

- 非零退出码
- 缺少pipeline_summary.json
- 任一阶段status不是success
- 指标中出现NaN或Inf
- Checkpoint不存在
- 发现test_used=true
- 发现分类输入泄漏
- 输出目录与已有正式实验冲突

Harness应保存失败trial的stdout、stderr、配置和输出目录，不得将失败结果参与候选比较。

## 7. Validation目标

Forecast主要比较：

- RMSE_macro_norm：越低越好
- MAE_macro_norm：越低越好

同时报告：

- RMSE_macro_mps
- MAE_macro_mps
- T+1、T+4、T+8分时距结果

Classification主要比较：

- macro_f1：越高越好
- CSI_macro：越高越好

同时报告：

- accuracy
- GOOD/PRECIP/HAZARD各类别F1
- GOOD/PRECIP/HAZARD各类别CSI
- confusion matrix
- zero-support状态

Harness只能根据训练集与验证集结果选择研究动作。

## 8. 分类泄漏限制

Classification模型输入禁止包含：

- weather_code_id
- weather_label
- significant_wx
- weather label的确定性派生变量

weather_label只能作为监督目标使用。

允许的分类外生输入：

- sky_condition
- has_gust
- is_cavok
- visibility
- cloud_height
- gust_speed

## 9. 数据与测试集边界

本实例的模型侧流程只允许访问冻结的：

- pretrain/train
- pretrain/val
- trainval/train
- trainval/val

Harness不得：

- 读取密封测试集
- 索引测试集目录
- 修改冻结数据划分
- 将验证集并入训练集
- 根据测试集指标选择模型
- 修改任务标签映射和正式指标定义

正式密封测试必须在候选方案冻结后，由独立Test Custodian执行。

## 10. 输出目录保护

每次Harness运行必须创建新的RUN_ID。

建议格式：

<task>_<timestamp>_<seed>_<short_hash>

示例：

trial_20260827_001_seed1001_a13f8c

控制器支持识别已成功阶段，但不得依赖覆盖旧目录恢复实验。若目录非空但阶段不完整，应保留现场并创建新RUN_ID重跑。

## 11. 当前Baseline结论

Seed 43验证结果：

- Forecast scratch优于当前pretrained方案
- Classification pretrained优于scratch方案

这说明预训练迁移效果具有任务依赖性：

- 对Classification存在正迁移
- 对Forecast当前配置下没有观察到正迁移

Harness后续可以围绕该差异提出和验证研究假设，但不能修改固定评测协议。

## 12. 交付范围

本次交付可用于：

- Harness接口联调
- 自动实验调度
- 训练配置搜索
- 模型结构消融
- 预训练与下游迁移研究
- 验证候选自动选择
- 实验日志和研究谱系记录

本次交付不包含：

- 密封测试集
- 测试集指标
- 最终论文多随机种子统计
- Test Custodian权限