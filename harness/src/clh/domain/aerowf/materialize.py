"""Index-backed tensor materialization when release npy files are not extracted.

Values stay inside DATA_CONTRACT_v1 frozen min/max ranges. This is a harness
fallback so the paper loop can run on the model-training index; it is not the
custodian 7z payload.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from clh.domain.aerowf.contract import (
    EXO_CONTINUOUS_CHANNELS,
    HAZARD_WIND_MPS,
    N_MAX,
    RUNWAY_CHANNELS,
    RUNWAY_COUNT,
    TIME_STEPS,
    WIND_X_INDEX,
    WIND_Y_INDEX,
)
from clh.domain.aerowf.frames import AeroFrame, last_step_wind_speed
from clh.domain.aerowf.io import IndexRow, apply_runway_minmax, invert_runway_minmax


CLIMATE = {
    "ZBAA": {"base": 4.8, "temp": 12.0, "vis": 5.6, "heading": 36.0},
    "ZSPD": {"base": 5.6, "temp": 16.0, "vis": 5.8, "heading": 16.0},
    "ZSSS": {"base": 5.1, "temp": 17.0, "vis": 5.9, "heading": 18.0},
    "ZBAD": {"base": 7.4, "temp": 11.0, "vis": 5.4, "heading": 1.0},
    "ZJHK": {"base": 4.2, "temp": 26.0, "vis": 6.0, "heading": 9.0},
}


def materialize_frame(
    rows: list[IndexRow],
    stats: dict,
    *,
    source: str,
    role: str,
    climate_override: str | None = None,
    seed: int = 17,
) -> AeroFrame:
    n = len(rows)
    if n == 0:
        from clh.domain.aerowf.frames import empty_frame

        frame = empty_frame(source=source)
        frame.role = role
        return frame

    airports = np.array([row.airport for row in rows], dtype=object)
    stamps = np.array([row.timestamp for row in rows], dtype="datetime64[s]")
    sample_id = np.array([row.sample_id for row in rows], dtype=object)
    hours = _hour_of_day(stamps)
    months = _month(stamps)

    raw = np.zeros((n, N_MAX, TIME_STEPS, RUNWAY_CHANNELS), dtype=np.float32)
    mask = np.zeros((n, N_MAX), dtype=bool)
    exo = np.zeros((n, EXO_CONTINUOUS_CHANNELS), dtype=np.float32)
    event_id = np.empty((n,), dtype=object)
    weather_label = np.zeros((n,), dtype=np.int64)
    rng = np.random.default_rng(seed)

    step = np.arange(TIME_STEPS, dtype=np.float32)
    for i, row in enumerate(rows):
        climate_key = climate_override or row.airport
        spec = CLIMATE.get(climate_key, CLIMATE["ZBAA"])
        n_runways = RUNWAY_COUNT.get(row.airport, 3)
        mask[i, :n_runways] = True
        hour = float(hours[i])
        month = float(months[i])
        storm = _is_storm(row.timestamp)
        event_id[i] = "storm" if storm else "none"
        t_grid = (hour - (TIME_STEPS - 1 - step) / 60.0) % 24.0
        diurnal = 3.2 * np.sin(2 * np.pi * t_grid / 24.0)
        noise = rng.normal(0.0, 0.9, TIME_STEPS).astype(np.float32)
        speed = np.clip(spec["base"] + diurnal + noise + (8.5 if storm else 0.0), 0.4, 28.0)
        heading = spec["heading"]
        direction = (220.0 + rng.normal(0.0, 18.0, TIME_STEPS) + (90.0 if climate_key == "ZJHK" else 0.0)) % 360.0
        rad = np.deg2rad(direction)
        wx = speed * np.sin(rad)
        wy = speed * np.cos(rad)
        # AR(1) on the last step so persistence is strong but not perfect.
        wx[-1] = 0.82 * wx[-2] + 0.18 * wx[-1] + 0.12 * np.cos(np.deg2rad(direction[-1] - heading))
        wy[-1] = 0.82 * wy[-2] + 0.18 * wy[-1]
        temp = spec["temp"] + 6.0 * np.sin(2 * np.pi * t_grid / 24.0) + rng.normal(0.0, 1.4, TIME_STEPS)
        raw[i, :, :, 0] = 1200.0 + 400.0 * np.sin(2 * np.pi * month / 12.0)
        raw[i, :, :, WIND_X_INDEX] = wx
        raw[i, :, :, WIND_Y_INDEX] = wy
        raw[i, :, :, 3] = 1013.0 + rng.normal(0.0, 3.0, TIME_STEPS)
        raw[i, :, :, 4] = temp
        raw[i, :, :, 5] = np.clip(55.0 + rng.normal(0.0, 8.0, TIME_STEPS), 5.0, 99.0)
        raw[i, :, :, 6] = temp - 4.0
        raw[i, :, :, 7] = np.sin(2 * np.pi * t_grid / 24.0)
        raw[i, :, :, 8] = np.cos(2 * np.pi * t_grid / 24.0)
        raw[i, :, :, 9] = np.sin(2 * np.pi * month / 12.0)
        raw[i, :, :, 10] = np.cos(2 * np.pi * month / 12.0)
        for slot in range(n_runways, N_MAX):
            raw[i, slot] = 0.0
        exo[i, 0] = spec["vis"] + rng.normal(0.0, 0.15)
        exo[i, 1] = np.log1p(1800.0 + rng.normal(0.0, 120.0))
        exo[i, 2] = 0.0
        weather_label[i] = int(storm) + int(speed[-1] >= HAZARD_WIND_MPS)

    runway = apply_runway_minmax(raw, stats)
    physical = invert_runway_minmax(runway, stats)
    prev = last_step_wind_speed(physical, mask, step=-2)
    storm = np.array([eid == "storm" for eid in event_id], dtype=float)
    y = (0.70 * prev + 0.11 * hours + 0.45 + 3.2 * storm).astype(np.float32)
    hazard = y >= HAZARD_WIND_MPS
    return AeroFrame(
        airports=airports,
        time=stamps,
        sample_id=sample_id,
        runway=runway,
        runway_mask=mask,
        exo_continuous=exo.astype(np.float32),
        y=y,
        hazard=hazard,
        wind_speed=y,
        prev_wind_speed=prev.astype(np.float32),
        hour=hours.astype(np.float32),
        event_id=event_id,
        weather_label=weather_label,
        hazard_threshold=HAZARD_WIND_MPS,
        source=source,
        role=role,
    )


def rows_from_stamps(
    airport: str,
    stamps: Iterable[np.datetime64],
    *,
    prefix: str,
) -> list[IndexRow]:
    rows: list[IndexRow] = []
    for i, stamp in enumerate(stamps):
        rows.append(
            IndexRow(
                sample_id=f"{prefix}:{airport}:{i:08d}",
                airport=airport,
                timestamp=np.datetime64(stamp, "s"),
                source_split="external",
                role="external",
                source_index=i,
            )
        )
    return rows


def _hour_of_day(stamps: np.ndarray) -> np.ndarray:
    minutes = stamps.astype("datetime64[m]").astype(np.int64)
    return ((minutes % (24 * 60)) / 60.0).astype(np.float32)


def _month(stamps: np.ndarray) -> np.ndarray:
    return np.array([int(str(s)[5:7]) for s in stamps], dtype=np.float32)


def _is_storm(stamp: np.datetime64) -> bool:
    hour = int(stamp.astype("datetime64[h]").astype(np.int64) % 24)
    day = int(stamp.astype("datetime64[D]").astype(np.int64) % 17)
    return hour in {15, 16, 17} and day in {2, 9, 14}
