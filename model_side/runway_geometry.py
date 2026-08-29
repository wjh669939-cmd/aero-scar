"""Runway geometry constants (B3 frozen delivery, train airports only).

来源：数据侧 `runway_headings.json`（SHA-256
5cfac91d352ac1b20d09b7625a171e9a0e407ea26d7d1bedc55689d542e29799），
状态 FROZEN_WITH_PROVENANCE_LIMITATION。语义与限制（原文如实转录）：

- heading 为跑道号名义朝向（designator ×10°），非测绘精确方位角；
- 槽位映射假设：released 张量的 runway 槽位顺序 == 原始 AWOS runways_data 的
  physical runway group 顺序（source_order_index）。上游生产管线不可得，该映射
  不可独立证明（provenance limitation，项目方已接受残余风险）；
- 本模块仅含训练/验证三机场；认证机场的同构表由评测侧在其私有环境按同一
  schema 提供（trial 代码经参数接收朝向，不得依赖机场身份）。

消费方：训练脚本的数据集层（按机场查表、按槽位对齐、虚拟槽位补 NaN），
经 build_*_inputs 的 `runway_axis_heading_deg` 参数传给 trial 代码。
trial 代码不应 import 本模块。
"""

from __future__ import annotations

import numpy as np

PHYSICAL_RUNWAY_COUNTS = {"ZBAA": 3, "ZSPD": 4, "ZSSS": 2}

# 轴向朝向（mod 180°），索引 = source_order_index
AXIS_HEADING_DEG_MOD180 = {
    "ZBAA": (0.0, 0.0, 10.0),
    "ZSPD": (170.0, 160.0, 170.0, 160.0),
    "ZSSS": (0.0, 0.0),
}


def slot_axis_headings(airport: str, n_slots: int) -> np.ndarray:
    """槽位对齐的轴向朝向数组；虚拟/越界槽位为 NaN。未知机场返回全 NaN
    （trial 代码必须容忍 NaN——虚拟槽位与朝向未知场景的统一语义）。"""
    out = np.full(n_slots, np.nan, dtype=np.float32)
    headings = AXIS_HEADING_DEG_MOD180.get(airport)
    if headings:
        k = min(len(headings), n_slots)
        out[:k] = headings[:k]
    return out
