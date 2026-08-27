"""Tensor sample batch for AeroWF Data Contract V1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clh.domain.aerowf.contract import HAZARD_WIND_MPS, N_MAX, TIME_STEPS, WIND_X_INDEX, WIND_Y_INDEX


@dataclass
class AeroFrame:
    """One evaluator-owned split. Tensors follow DATA_CONTRACT_v1.md."""

    airports: np.ndarray
    time: np.ndarray
    sample_id: np.ndarray
    runway: np.ndarray
    runway_mask: np.ndarray
    exo_continuous: np.ndarray
    y: np.ndarray
    hazard: np.ndarray
    wind_speed: np.ndarray
    prev_wind_speed: np.ndarray
    hour: np.ndarray
    event_id: np.ndarray
    weather_label: np.ndarray | None = None
    exo_categorical: np.ndarray | None = None
    hazard_threshold: float = HAZARD_WIND_MPS
    source: str = "trainval"
    role: str = "trainval_train"
    stats_name: str = "train_stats_v1.json"

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def identities(self) -> set[tuple[str, str]]:
        return {(str(a), str(s)) for a, s in zip(self.airports, self.sample_id)}

    def time_identities(self) -> set[tuple[str, np.datetime64]]:
        stamps = _as_datetime64(self.time)
        return {(str(a), stamps[i]) for i, a in enumerate(self.airports)}


def empty_frame(*, source: str = "empty") -> AeroFrame:
    zero = np.zeros((0,), dtype=np.int64)
    return AeroFrame(
        airports=np.zeros((0,), dtype=object),
        time=np.array([], dtype="datetime64[s]"),
        sample_id=np.zeros((0,), dtype=object),
        runway=np.zeros((0, N_MAX, TIME_STEPS, 11), dtype=np.float32),
        runway_mask=np.zeros((0, N_MAX), dtype=bool),
        exo_continuous=np.zeros((0, 3), dtype=np.float32),
        y=np.zeros((0,), dtype=np.float32),
        hazard=np.zeros((0,), dtype=bool),
        wind_speed=np.zeros((0,), dtype=np.float32),
        prev_wind_speed=np.zeros((0,), dtype=np.float32),
        hour=np.zeros((0,), dtype=np.float32),
        event_id=np.zeros((0,), dtype=object),
        source=source,
    )


def concat_frames(frames: list[AeroFrame]) -> AeroFrame:
    if not frames:
        raise ValueError("no frames to concatenate")
    first = frames[0]
    weather_label = None
    if all(f.weather_label is not None for f in frames):
        weather_label = np.concatenate([f.weather_label for f in frames])  # type: ignore[arg-type]
    exo_cat = None
    if all(f.exo_categorical is not None for f in frames):
        exo_cat = np.concatenate([f.exo_categorical for f in frames])  # type: ignore[arg-type]
    return AeroFrame(
        airports=np.concatenate([f.airports for f in frames]),
        time=np.concatenate([_as_datetime64(f.time) for f in frames]),
        sample_id=np.concatenate([f.sample_id for f in frames]),
        runway=np.concatenate([f.runway for f in frames], axis=0),
        runway_mask=np.concatenate([f.runway_mask for f in frames], axis=0),
        exo_continuous=np.concatenate([f.exo_continuous for f in frames], axis=0),
        y=np.concatenate([f.y for f in frames]),
        hazard=np.concatenate([f.hazard for f in frames]),
        wind_speed=np.concatenate([f.wind_speed for f in frames]),
        prev_wind_speed=np.concatenate([f.prev_wind_speed for f in frames]),
        hour=np.concatenate([f.hour for f in frames]),
        event_id=np.concatenate([f.event_id for f in frames]),
        weather_label=weather_label,
        exo_categorical=exo_cat,
        hazard_threshold=first.hazard_threshold,
        source="+".join(sorted({f.source for f in frames})),
        role=first.role,
        stats_name=first.stats_name,
    )


def mask_frame(frame: AeroFrame, keep: np.ndarray) -> AeroFrame:
    keep = np.asarray(keep, dtype=bool)
    weather_label = frame.weather_label[keep] if frame.weather_label is not None else None
    exo_cat = frame.exo_categorical[keep] if frame.exo_categorical is not None else None
    return AeroFrame(
        airports=frame.airports[keep],
        time=_as_datetime64(frame.time)[keep],
        sample_id=frame.sample_id[keep],
        runway=frame.runway[keep],
        runway_mask=frame.runway_mask[keep],
        exo_continuous=frame.exo_continuous[keep],
        y=frame.y[keep],
        hazard=frame.hazard[keep],
        wind_speed=frame.wind_speed[keep],
        prev_wind_speed=frame.prev_wind_speed[keep],
        hour=frame.hour[keep],
        event_id=frame.event_id[keep],
        weather_label=weather_label,
        exo_categorical=exo_cat,
        hazard_threshold=frame.hazard_threshold,
        source=frame.source,
        role=frame.role,
        stats_name=frame.stats_name,
    )


def last_step_wind_speed(runway: np.ndarray, runway_mask: np.ndarray, step: int = -1) -> np.ndarray:
    """Mask-aware pooled wind speed at one window step. Never infers validity from values."""
    wx = runway[:, :, step, WIND_X_INDEX]
    wy = runway[:, :, step, WIND_Y_INDEX]
    speed = np.hypot(wx, wy)
    weights = runway_mask.astype(np.float32)
    denom = np.maximum(weights.sum(axis=1), 1.0)
    return (speed * weights).sum(axis=1) / denom


def _as_datetime64(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[s]")
    return np.asarray(values, dtype="datetime64[s]")
