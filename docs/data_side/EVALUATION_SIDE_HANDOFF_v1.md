# AeroWF v1 --- 评测侧交付说明

## 1. 本次接收文件

评测侧请接收并使用：

-   `AeroWF_v1_EVALUATION.7z`
-   `DELIVERY_ARCHIVES_SHA256.txt`
-   本说明文件 `EVALUATION_SIDE_HANDOFF_v1.md`

评测材料请与模型训练侧保持隔离；不要将评测包内容转发给模型训练人员。

## 2. 用途与边界

`AeroWF_v1_EVALUATION.7z` 为独立评测侧正式冻结交付包。

请遵守以下边界：

-   评测包仅用于既定评测流程。
-   不得将评测数据、标签、统计结果或可反推出评测内容的信息提供给模型训练侧。
-   不得使用评测数据反向参与训练、调参、特征工程或人工规则设计。
-   评测前请保持交付包原样；如发现缺失、损坏或口径疑问，请先联系数据交付方确认。
-   需要留档时，建议记录所使用压缩包的
    SHA-256，以保证评测结果与数据版本可追溯。

## 3. 完整性校验

交付前已完成最终 SHA-256 校验。

收到文件后，请使用 `DELIVERY_ARCHIVES_SHA256.txt` 对
`AeroWF_v1_EVALUATION.7z` 再做一次 SHA-256 核验。

PowerShell 示例：

``` powershell
Get-FileHash .\AeroWF_v1_EVALUATION.7z -Algorithm SHA256
```

输出 Hash 应与 `DELIVERY_ARCHIVES_SHA256.txt` 中对应条目完全一致。

## 4. 建议回执

核验完成后请回复：

> 评测侧已收到 AeroWF v1 EVALUATION 交付包，SHA-256
> 校验通过，将按独立评测要求保持数据隔离并开展评测。
