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


_R6_STEPS = [4]
_R6_WINDOW = 8


def _multiscale_diff(runway_arr: np.ndarray) -> np.ndarray:
    """R6: 多尺度差分按注册语义实现为追加通道（结构可行性由闸门裁决）。"""
    arr = np.array(runway_arr, dtype=np.float32, copy=True)
    extras = []
    for step in _R6_STEPS:
        d = np.zeros_like(arr[..., 1:3])
        d[:, step:, :] = arr[:, step:, 1:3] - arr[:, :-step, 1:3]
        extras.append(d)
    if _R6_WINDOW:
        w = _R6_WINDOW
        var = np.zeros_like(arr[..., 1:3])
        for t in range(arr.shape[1]):
            lo = max(0, t - w + 1)
            var[:, t, :] = arr[:, lo:t + 1, 1:3].var(axis=1)
        extras.append(var)
    return np.concatenate([arr] + extras, axis=-1)

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
    del runway_axis_heading_deg  # baseline ignores runway geometry
    return {
        "x": torch.from_numpy(_multiscale_diff(runway_arr)),
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
    del runway_axis_heading_deg  # baseline ignores runway geometry
    return {
        "x": torch.from_numpy(_multiscale_diff(runway_arr)),
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
