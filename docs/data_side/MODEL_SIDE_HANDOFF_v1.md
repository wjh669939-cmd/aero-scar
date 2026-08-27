# AeroWF v1 --- 模型侧交付说明

## 1. 本次接收文件

模型侧请接收并使用：

-   `AeroWF_v1_MODEL_TRAINING.7z`
-   `DELIVERY_ARCHIVES_SHA256.txt`
-   本说明文件 `MODEL_SIDE_HANDOFF_v1.md`

如需审计或复现材料，请联系数据交付方，不要自行使用评测侧交付包。

## 2. 用途与边界

`AeroWF_v1_MODEL_TRAINING.7z` 为模型训练侧正式冻结交付包。

请遵守以下边界：

-   自监督预训练仅使用 `release_v1/pretrain/train` 进行优化。
-   `release_v1/pretrain/val` 仅用于验证。
-   `release_v1/pretrain/test` 保持 held-out，不参与训练拟合。
-   PRE2020 的 `weather_label.npy` 不得作为自监督学习目标使用。
-   不得使用评测侧/密封评测数据进行训练、调参、特征工程或人工规则设计。
-   如需修改数据处理逻辑，请先与数据交付方确认，不要直接修改冻结交付包。

## 3. 完整性校验

交付前已完成最终 SHA-256 校验。

收到文件后，请使用 `DELIVERY_ARCHIVES_SHA256.txt` 对
`AeroWF_v1_MODEL_TRAINING.7z` 再做一次 SHA-256 核验。

PowerShell 示例：

``` powershell
Get-FileHash .\AeroWF_v1_MODEL_TRAINING.7z -Algorithm SHA256
```

输出 Hash 应与 `DELIVERY_ARCHIVES_SHA256.txt` 中对应条目完全一致。

## 4. 建议回执

核验完成后请回复：

> 模型侧已收到 AeroWF v1 MODEL_TRAINING 交付包，SHA-256
> 校验通过，后续将按数据边界开展训练。
