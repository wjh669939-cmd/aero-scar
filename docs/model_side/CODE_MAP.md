这几个问题可以明确回答，而且应当加入交付说明。当前代码不是把所有逻辑都放在一个文件里，而是分为：

- 任务输入与标签构造
- AeroWF 公共编码器
- 预训练损失
- 下游损失
- 训练循环与优化器
- 全流程调度器

## 一、代码位置总表

| 模块 | 主要文件 | 核心类/函数 |
|---|---|---|
| 预训练输入构造 | `aerowf_unified_pretrain_train_v2.py` | `FrozenAeroWFDataset` |
| Forecast 输入与目标构造 | `aerowf_forecast_train_v2.py` | `AirportForecastDataset` |
| Forecast 模型头 | `aerowf_forecast_train_v2.py` | `AeroWFForecastModel` |
| Classification 输入与标签映射 | `aerowf_classification_train_v2.py` | `map_labels`、`AirportClassificationDataset` |
| Classification 安全外生编码 | `aerowf_classification_train_v2.py` | `AllowedContextEncoder` |
| Classification 模型头 | `aerowf_classification_train_v2.py` | `AeroWFClassificationModel` |
| AeroWF 公共时域/频域编码 | `models/AirFM/unified_model.py` | `UnifiedSeries2Vec.encode` |
| 频域编码器 | `models/AirFM/encoders/frets_encoder.py` | FreTS 相关编码类 |
| Forecast 损失 | `aerowf_forecast_train_v2.py` | `masked_mse` |
| Classification 损失 | `aerowf_classification_train_v2.py` | `balanced_class_weights`、`F.cross_entropy` |
| 预训练联合损失组装 | `models/AirFM/unified_model.py` | `unified_pretrain_forward` |
| 重建损失 | `models/AirFM/masked.py` | `masked_mse_loss` |
| 时间/频谱物理距离 | `models/AirFM/physics_distance.py` | `PhysicsDistanceComputer` |
| SoftDTW 实现 | `models/AirFM/soft_dtw_cuda.py` | `SoftDTW` |
| 频率过滤/频谱处理 | `models/AirFM/fft_filter.py` | `filter_frequencies` |
| 预训练循环 | `models/AirFM/unified_trainer.py` | `train`、`train_epoch`、`_unified_pretrain_step`、`validate` |
| 预训练 Optimizer 配置 | `models/AirFM/unified_trainer.py` | `_create_optimizer` |
| Forecast 训练循环与 Optimizer | `aerowf_forecast_train_v2.py` | `train_epoch`、`evaluate`、`main` |
| Classification 训练循环与 Optimizer | `aerowf_classification_train_v2.py` | `train_epoch`、`evaluate`、`main` |
| 五阶段流程调度 | `aerowf_full_pipeline_v2.py` | `run_stage`、`main` |

## 二、特征与输入编码具体在哪里

### 1. 预训练输入

文件：

```text
/root/autodl-tmp/aerowf_baseline/AeroWF/aerowf_unified_pretrain_train_v2.py
```

核心类：

```python
FrozenAeroWFDataset
```

负责读取和构造：

- `runway.npy`：`[B,4,96,11]`
- `runway_mask.npy`：真实跑道掩码
- 外生分类变量
- 外生连续变量
- 训练/验证 DataLoader
- 不使用 `weather_label` 作为预训练目标

真正的 AeroWF 编码发生在：

```text
models/AirFM/unified_model.py
```

核心方法：

```python
UnifiedSeries2Vec.encode(...)
```

这里组合：

- `encoder_T`：时域编码
- `encoder_F`：频域编码
- GNN/跑道节点交互
- hierarchy
- exogenous encoder
- temporal/frequency fusion

### 2. Forecast 输入和目标

文件：

```text
/root/autodl-tmp/aerowf_downstream_v2/src/aerowf_forecast_train_v2.py
```

输入构造：

```python
AirportForecastDataset
```

它通过时间戳严格生成：

- T+1：窗口结束后15分钟
- T+4：窗口结束后60分钟
- T+8：窗口结束后120分钟

预测目标为每条真实跑道的：

```text
wind_x
wind_y
```

目标形状：

```text
[4, 3, 2]
```

模型封装：

```python
AeroWFForecastModel
```

调用公共编码器：

```python
self.core.encode(...)
```

然后使用：

```python
self.forecast_head
```

输出每条跑道、三个时距、两个风矢量分量。

### 3. Classification 输入和标签

文件：

```text
/root/autodl-tmp/aerowf_downstream_v2/src/aerowf_classification_train_v2.py
```

标签映射：

```python
map_labels(...)
```

数据输入：

```python
AirportClassificationDataset
```

安全外生特征编码：

```python
AllowedContextEncoder
```

分类模型：

```python
AeroWFClassificationModel
```

分类阶段明确不加载：

- `weather_code_id`
- `significant_wx`

并且：

- `weather_label` 只作为监督目标
- 原 AeroWF `exo_encoder` 被绕过
- 只使用协议允许的外生变量

## 三、下游损失定义在哪里

### Forecast 损失

文件：

```text
aerowf_forecast_train_v2.py
```

函数：

```python
masked_mse(...)
```

核心计算：

```python
mask = node_mask[:, :, None, None].expand_as(prediction)
loss = torch.square(prediction - target)[mask].mean()
```

也就是说：

- 使用 MSE
- 只计算真实跑道
- 填充出来的虚拟跑道不参与损失
- 覆盖 T+1/T+4/T+8 和 wind_x/wind_y

训练调用位置：

```python
train_epoch(...)
```

### Classification 损失

文件：

```text
aerowf_classification_train_v2.py
```

类别权重：

```python
train_label_counts(...)
balanced_class_weights(...)
```

损失：

```python
F.cross_entropy(
    logits,
    label,
    weight=class_weights,
    ignore_index=IGNORE_INDEX,
)
```

当前使用：

- 训练集统计得到的平衡类别权重
- Weighted Cross Entropy
- `IGNORE_INDEX=-100`
- 0/1 类上游保留标签被忽略
- 不使用验证集统计类别权重

模型选择依据不是 accuracy，而是验证集：

```text
Macro-F1
```

## 四、预训练 SoftDTW 和谱对齐在哪里

预训练损失不是只在一个文件里，分为三层。

### 1. 联合损失组装

文件：

```text
models/AirFM/unified_model.py
```

函数：

```python
unified_pretrain_forward(...)
```

总体形式：

\[
L =
\lambda_{\text{recon}} L_{\text{recon}}
+
\lambda_{\text{contrast}}
\left(L_T+L_F\right)
\]

当前正式配置：

```text
lambda_recon = 1.0
lambda_contrast = 0.5
```

其中：

- `L_recon`：掩码重建损失
- `L_T`：时域物理距离对齐
- `L_F`：频域/频谱距离对齐

### 2. SoftDTW 与物理距离目标

文件：

```text
models/AirFM/physics_distance.py
```

核心类：

```python
PhysicsDistanceComputer
```

主要负责：

- 计算样本之间的时域物理距离
- 调用 SoftDTW
- 计算频域物理距离
- 维护距离范围的 EMA 归一化
- 生成表示空间对齐使用的目标距离矩阵
- 忽略虚拟跑道节点

SoftDTW CUDA 的具体算法实现位于：

```text
models/AirFM/soft_dtw_cuda.py
```

核心类：

```python
SoftDTW
```

### 3. 谱对齐

频域距离和频谱对齐的主要逻辑位于：

```text
models/AirFM/physics_distance.py
```

频域编码本身位于：

```text
models/AirFM/encoders/frets_encoder.py
```

频率过滤辅助实现位于：

```text
models/AirFM/fft_filter.py
```

因此可以向组员概括为：

```text
unified_model.py负责组装联合损失；
physics_distance.py负责构造SoftDTW和频谱物理距离目标；
soft_dtw_cuda.py负责SoftDTW底层CUDA计算；
frets_encoder.py负责频域表示编码。
```

### 4. 重建损失

文件：

```text
models/AirFM/masked.py
```

核心内容：

```python
generate_hybrid_mask(...)
apply_mask(...)
ReconstructionDecoder
masked_mse_loss(...)
```

负责：

- Random + Causal mask
- 25% mask ratio
- 节点级重建
- 只在被遮挡时间位置计算 MSE
- 过滤虚拟跑道

## 五、训练循环和 Optimizer 在哪里

### 预训练

入口脚本：

```text
aerowf_unified_pretrain_train_v2.py
```

它负责：

- 解析命令行参数
- 创建 Dataset/DataLoader
- 构造模型配置
- 实例化 `UnifiedSeries2Vec`
- 实例化 `UnifiedTrainer`
- 调用 `trainer.train(...)`

实际训练框架：

```text
models/AirFM/unified_trainer.py
```

核心函数：

```python
_create_optimizer(...)
train_epoch(...)
_unified_pretrain_step(...)
validate(...)
train(...)
```

当前正式预训练 Optimizer：

```text
AdamW
lr = 3e-4
weight_decay = 1e-4
warmup_epochs = 5
```

### Forecast

文件：

```text
aerowf_forecast_train_v2.py
```

训练循环：

```python
train_epoch(...)
```

验证：

```python
evaluate(...)
```

Optimizer 在：

```python
main(...)
```

根据初始化方式设置 AdamW 参数组：

- Scratch：core 和 head 都使用 `scratch_lr`
- Pretrained：encoder 使用 `encoder_lr`
- Pretrained：forecast head 使用 `head_lr`

### Classification

文件：

```text
aerowf_classification_train_v2.py
```

训练循环：

```python
train_epoch(...)
```

验证：

```python
evaluate(...)
```

Optimizer 在：

```python
main(...)
```

参数组：

- Scratch：core、context encoder、head 使用 `scratch_lr`
- Pretrained：core 使用较小的 `encoder_lr`
- Pretrained：context encoder 和 classification head 使用 `head_lr`

### 全流程控制器

文件：

```text
aerowf_full_pipeline_v2.py
```

函数：

```python
run_stage(...)
main(...)
```

它只负责：

- 调用五个实验阶段
- 检查退出码
- 检查 `metrics.json`
- 传递预训练 Checkpoint
- 汇总指标
- 选择验证候选

它本身不定义网络、损失或 Optimizer。

## 六、让组员快速定位对应代码

组员可以在目标实例执行：

```bash
cd /root/autodl-tmp/aerowf_baseline/AeroWF

grep -nE \
  "class FrozenAeroWFDataset|def encode|def unified_pretrain_forward|def masked_recon_forward" \
  aerowf_unified_pretrain_train_v2.py \
  models/AirFM/unified_model.py

grep -nE \
  "class PhysicsDistanceComputer|def compute_temporal_distance|def compute_frequency_distance|def forward" \
  models/AirFM/physics_distance.py

grep -nE \
  "class SoftDTW|def generate_hybrid_mask|def masked_mse_loss|class ReconstructionDecoder" \
  models/AirFM/soft_dtw_cuda.py \
  models/AirFM/masked.py

grep -nE \
  "def _create_optimizer|def train_epoch|def _unified_pretrain_step|def validate|def train" \
  models/AirFM/unified_trainer.py
```

下游任务：

```bash
grep -nE \
  "class AirportForecastDataset|class AeroWFForecastModel|def masked_mse|def train_epoch|def evaluate|AdamW" \
  /root/autodl-tmp/aerowf_downstream_v2/src/aerowf_forecast_train_v2.py

grep -nE \
  "def map_labels|class AirportClassificationDataset|class AllowedContextEncoder|class AeroWFClassificationModel|def balanced_class_weights|def train_epoch|cross_entropy|AdamW" \
  /root/autodl-tmp/aerowf_downstream_v2/src/aerowf_classification_train_v2.py
```

这部分最好单独保存成：

```text
CODE_MAP.md
```

加入交付包。由于完整数据已经传输完，后续只需要增量同步这个小文档，不需要重新传输 5.94GB。
