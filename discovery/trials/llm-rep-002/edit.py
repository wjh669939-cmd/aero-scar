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


def build_forecast_inputs(
    runway_arr: np.ndarray,
    runway_mask: np.ndarray,
    exo_cat: dict[str, int],
    exo_cont: np.ndarray,
    *,
    norm_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del norm_stats  # current baseline does not renormalize inputs
    # exo_cont layout (locked caller): [visibility, cloud_height, gust_speed,
    #   dewpoint_depression, pressure_trend_3h, gust_minus_mean]
    # The first three are the original fields; the last three are the new
    # physical precursor features added by this trial.
    # Handle both 3-element (legacy) and 6-element (full) arrays safely.
    exo_cont = np.asarray(exo_cont, dtype=np.float32).reshape(-1)
    if exo_cont.size == 3:
        # Legacy caller: only visibility, cloud_height, gust_speed provided.
        # Fill the three new precursor features with zeros.
        exo_cont = np.concatenate([exo_cont, np.zeros(3, dtype=np.float32)])
    elif exo_cont.size != 6:
        raise ValueError(
            f"exo_cont must have 3 or 6 elements, got {exo_cont.size}"
        )
    return {
        "x": torch.from_numpy(np.array(runway_arr, dtype=np.float32, copy=True)),
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
            "dewpoint_depression": torch.tensor(
                float(exo_cont[3]), dtype=torch.float32
            ),
            "pressure_trend_3h": torch.tensor(
                float(exo_cont[4]), dtype=torch.float32
            ),
            "gust_minus_mean": torch.tensor(
                float(exo_cont[5]), dtype=torch.float32
            ),
        },
    }


def build_classification_inputs(
    runway_arr: np.ndarray,
    runway_mask: np.ndarray,
    exo_cat: dict[str, int],
    exo_cont: np.ndarray,
) -> dict[str, Any]:
    return {
        "x": torch.from_numpy(np.array(runway_arr, dtype=np.float32, copy=True)),
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
