"""R-axis trial surface: how downstream inputs are represented.

Target generation, timestamp alignment, and label mapping stay in the locked
train scripts.  This file must not import those scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


FORBIDDEN_INPUT_COLUMNS = ("weather_code_id", "weather_label", "significant_wx")


_R4_COMPONENTS = ['gust_crosswind']


def _runway_frame(runway_arr: np.ndarray, heading_deg) -> np.ndarray:
    """R4: 把 wind_x/wind_y(ch1/ch2) 旋转到跑道轴坐标系（顺风/侧风分量）。

    heading 为 mod-180 名义朝向（度）；NaN 槽位保持原始分量（容忍语义）。
    分量选择: headwind -> ch1, crosswind -> ch2；只选其一时另一通道保留原值。
    gust_crosswind 依赖阵风矢量数据（接口未提供），按注册参数如实实现为
    crosswind 同值（阵风标量不可分解），并在此注明近似。
    """
    arr = np.array(runway_arr, dtype=np.float32, copy=True)
    if heading_deg is None:
        return arr
    theta = np.deg2rad(np.asarray(heading_deg, dtype=np.float32))  # (n_slots,)
    valid = np.isfinite(theta)
    if not valid.any():
        return arr
    u = arr[..., 1] - 0.5
    v = arr[..., 2] - 0.5
    sin_t = np.sin(theta)[:, None]
    cos_t = np.cos(theta)[:, None]
    headwind = u * sin_t + v * cos_t
    crosswind = -u * cos_t + v * sin_t
    vmask = valid[:, None]
    if "headwind" in _R4_COMPONENTS:
        arr[..., 1] = np.where(vmask, np.clip(headwind + 0.5, 0.0, 1.0), arr[..., 1])
    if "crosswind" in _R4_COMPONENTS or "gust_crosswind" in _R4_COMPONENTS:
        arr[..., 2] = np.where(vmask, np.clip(crosswind + 0.5, 0.0, 1.0), arr[..., 2])
    return arr

def build_forecast_inputs(
    runway_arr: np.ndarray,
    runway_mask: np.ndarray,
    exo_cat: dict[str, int],
    exo_cont: np.ndarray,
    *,
    norm_stats: dict[str, Any] | None = None,
    runway_axis_heading_deg: np.ndarray | None = None,
) -> dict[str, Any]:
    """runway_axis_heading_deg: (n_slots,) float32，跑道轴向名义朝向（mod 180°），
    虚拟槽位/未知机场为 NaN（B3 冻结交付，含槽位映射 provenance 限制——见
    runway_geometry 模块说明）。基线不使用；坐标系类变换可消费，必须容忍 NaN。"""
    del norm_stats  # current baseline does not renormalize inputs
    return {
        "x": torch.from_numpy(_runway_frame(runway_arr, runway_axis_heading_deg)),
        "node_mask": torch.from_numpy(np.array(runway_mask, dtype=bool, copy=True)),
        "exo_categorical": {
            "significant_wx": torch.tensor(
                int(exo_cat["weather_code"]) != 2, dtype=torch.long
            ),
            "sky_condition": torch.tensor(
                int(exo_cat["sky_condition"]), dtype=torch.long
            ),
            "has_gust": torch.tensor(int(exo_cat["has_gust"]), dtype=torch.long),
            "is_cavok": torch.tensor(int(exo_cat["is_cavok"]), dtype=torch.long),
        },
        "exo_continuous": {
            "visibility": torch.tensor(float(exo_cont[0]), dtype=torch.float32),
            "cloud_height": torch.tensor(float(exo_cont[1]), dtype=torch.float32),
            "gust_speed": torch.tensor(float(exo_cont[2]), dtype=torch.float32),
        },
    }


def build_classification_inputs(
    runway_arr: np.ndarray,
    runway_mask: np.ndarray,
    exo_cat: dict[str, int],
    exo_cont: np.ndarray,
    *,
    runway_axis_heading_deg: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "x": torch.from_numpy(_runway_frame(runway_arr, runway_axis_heading_deg)),
        "node_mask": torch.from_numpy(np.array(runway_mask, dtype=bool, copy=True)),
        "allowed_categorical": {
            "sky_condition": torch.tensor(
                int(exo_cat["sky_condition"]), dtype=torch.long
            ),
            "has_gust": torch.tensor(int(exo_cat["has_gust"]), dtype=torch.long),
            "is_cavok": torch.tensor(int(exo_cat["is_cavok"]), dtype=torch.long),
        },
        "exo_continuous": torch.from_numpy(
            np.array(exo_cont, dtype=np.float32, copy=True)
        ),
    }


class AllowedContextEncoder(nn.Module):
    """Encode only sky/gust/CAVOK and the three released continuous fields."""

    def __init__(self, sky_known_max: int, output_dim: int = 32):
        super().__init__()
        self.sky_known_max = int(sky_known_max)
        self.sky_unknown_id = self.sky_known_max + 1
        self.sky_embedding = nn.Embedding(self.sky_unknown_id + 1, 8)
        self.gust_embedding = nn.Embedding(2, 4)
        self.cavok_embedding = nn.Embedding(2, 4)
        self.network = nn.Sequential(
            nn.Linear(8 + 4 + 4 + 3, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        categorical: dict[str, torch.Tensor],
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        sky = categorical["sky_condition"]
        sky = torch.where(
            (sky >= 0) & (sky <= self.sky_known_max),
            sky,
            torch.full_like(sky, self.sky_unknown_id),
        )
        gust = categorical["has_gust"].clamp(0, 1)
        cavok = categorical["is_cavok"].clamp(0, 1)
        features = torch.cat(
            [
                self.sky_embedding(sky),
                self.gust_embedding(gust),
                self.cavok_embedding(cavok),
                continuous,
            ],
            dim=-1,
        )
        return self.network(features)
