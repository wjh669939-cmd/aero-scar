"""AeroWF Data Contract V1 constants and leakage boundaries.

Source of truth: `任务测试/AeroWF/数据接口/AeroWF_v1_MODEL_TRAINING/`.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final = "1.0"
N_MAX: Final = 4
TIME_STEPS: Final = 96
RUNWAY_CHANNELS: Final = 11
EXO_CONTINUOUS_CHANNELS: Final = 3
EXO_CATEGORICAL_CHANNELS: Final = 4

RUNWAY_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "cloud_base",
    "wind_x",
    "wind_y",
    "pressure",
    "temperature",
    "humidity",
    "dewpoint",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
)

EXO_CONTINUOUS_NAMES: Final[tuple[str, ...]] = (
    "visibility",
    "cloud_height",
    "gust_speed",
)

EXO_CATEGORICAL_NAMES: Final[tuple[str, ...]] = (
    "weather_code_id",
    "sky_condition",
    "has_gust",
    "is_cavok",
)

WIND_X_INDEX: Final = 1
WIND_Y_INDEX: Final = 2
TEMPERATURE_INDEX: Final = 4
HOUR_SIN_INDEX: Final = 7
HOUR_COS_INDEX: Final = 8

TRAINING_AIRPORTS: Final[tuple[str, ...]] = ("ZBAA", "ZSPD", "ZSSS")
SPATIAL_HOLDOUT_AIRPORT: Final = "ZBAD"

# MODEL_HANDOFF_v1.md §2 / §3
ALLOWED_SEARCH_ROLES: Final[frozenset[str]] = frozenset(
    {
        "pretrain/train",
        "pretrain/val",
        "trainval/train",
        "trainval/val",
    }
)
FORBIDDEN_SEARCH_TOKENS: Final[tuple[str, ...]] = (
    "sealed",
    "ZBAD",
    "pretrain/test",
    "pretrain\\test",
)

TENSOR_FILES: Final[tuple[str, ...]] = (
    "runway.npy",
    "runway_mask.npy",
    "exo_continuous.npy",
    "weather_label.npy",
)

DOWNSTREAM_STATS_NAME: Final = "train_stats_v1.json"
PRETRAIN_STATS_NAME: Final = "pretrain_stats_v1.json"

# Paper §3.7 same-source threshold, mapped in 04-axis-data.md
SAME_SOURCE_RATE: Final = 0.05
# 96-minute lookback + 15-minute sample cadence (DATA_CONTRACT / RELEASE_NOTES)
NEAR_ANALOGUE_MINUTES: Final = 111
HAZARD_WIND_MPS: Final = 12.0

RUNWAY_COUNT: Final[dict[str, int]] = {
    "ZBAA": 3,
    "ZSPD": 4,
    "ZSSS": 2,
    "ZBAD": 2,
}
