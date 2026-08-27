"""Synthetic aerodrome weather used by the dummy domain and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SplitName = Literal["train", "val", "test_temporal", "test_spatial", "test_event"]
HAZARD_THRESHOLD = 12.0
SOURCE_AIRPORTS = ("ZBAA", "ZSPD", "ZGGG")
HOLDOUT_AIRPORT = "ZUUU"
STORM_RANGES = ((80, 95), (310, 325))


@dataclass
class WeatherFrame:
    airports: np.ndarray
    time: np.ndarray
    wind_speed: np.ndarray
    wind_dir: np.ndarray
    temp: np.ndarray
    vis: np.ndarray
    hour: np.ndarray
    runway_heading: np.ndarray
    event_id: np.ndarray
    y: np.ndarray
    hazard: np.ndarray
    hazard_threshold: float = HAZARD_THRESHOLD
    source: str = "internal"

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def identities(self) -> set[tuple[str, int]]:
        return {(str(a), int(t)) for a, t in zip(self.airports, self.time)}


def concat_frames(frames: list[WeatherFrame]) -> WeatherFrame:
    if not frames:
        raise ValueError("no frames to concatenate")
    first = frames[0]
    return WeatherFrame(
        airports=np.concatenate([f.airports for f in frames]),
        time=np.concatenate([f.time for f in frames]),
        wind_speed=np.concatenate([f.wind_speed for f in frames]),
        wind_dir=np.concatenate([f.wind_dir for f in frames]),
        temp=np.concatenate([f.temp for f in frames]),
        vis=np.concatenate([f.vis for f in frames]),
        hour=np.concatenate([f.hour for f in frames]),
        runway_heading=np.concatenate([f.runway_heading for f in frames]),
        event_id=np.concatenate([f.event_id for f in frames]),
        y=np.concatenate([f.y for f in frames]),
        hazard=np.concatenate([f.hazard for f in frames]),
        hazard_threshold=first.hazard_threshold,
        source="+".join(sorted({f.source for f in frames})),
    )


@dataclass
class SyntheticAirportWeather:
    seed: int
    source_airports: tuple[str, ...]
    holdout_airports: tuple[str, ...]
    frames: dict[str, WeatherFrame]


def build_weather(
    *,
    seed: int = 7,
    source_airports: tuple[str, ...] = SOURCE_AIRPORTS,
    holdout_airports: tuple[str, ...] = (HOLDOUT_AIRPORT,),
) -> SyntheticAirportWeather:
    rng = np.random.default_rng(seed)
    hours = np.arange(400)
    storm = _storm_mask(hours)
    source_rows: list[WeatherFrame] = []
    for airport in source_airports:
        source_rows.append(_simulate_airport(rng, airport, hours, climate="temperate", storm=storm))
    source = concat_frames(source_rows)
    holdout = concat_frames(
        [_simulate_airport(rng, airport, hours, climate="plateau", storm=storm) for airport in holdout_airports]
    )
    frames = {
        "train": _select(source, (source.time < 240) & ~_is_storm(source)),
        "val": _select(source, (source.time >= 240) & (source.time < 300) & ~_is_storm(source)),
        "test_temporal": _select(source, (source.time >= 300) & ~_is_storm(source)),
        "test_spatial": _select(holdout, (holdout.time >= 240) & (holdout.time < 300) & ~_is_storm(holdout)),
        "test_event": _select(source, _is_storm(source)),
        "matched_ZBHH": _simulate_airport(rng, "ZBHH", hours[hours < 240], climate="temperate", storm=_storm_mask(hours[hours < 240])),
        "shifted_ZJHK": _simulate_airport(rng, "ZJHK", hours[hours < 240], climate="tropical", storm=_storm_mask(hours[hours < 240])),
        "leak_ZBAA_future": _select(source, (source.airports == "ZBAA") & (source.time >= 240)),
    }
    return SyntheticAirportWeather(
        seed=seed,
        source_airports=source_airports,
        holdout_airports=holdout_airports,
        frames=frames,
    )


def load_split(weather: SyntheticAirportWeather, split: SplitName) -> WeatherFrame:
    return weather.frames[split]


def hidden_identity_sets(weather: SyntheticAirportWeather) -> dict[str, set[tuple[str, int]]]:
    return {
        name: weather.frames[name].identities()
        for name in ("val", "test_temporal", "test_spatial", "test_event")
    }


def _storm_mask(hours: np.ndarray) -> np.ndarray:
    mask = np.zeros(hours.shape, dtype=bool)
    for start, end in STORM_RANGES:
        mask |= (hours >= start) & (hours < end)
    return mask


def _is_storm(frame: WeatherFrame) -> np.ndarray:
    return frame.event_id != "none"


def _select(frame: WeatherFrame, mask: np.ndarray) -> WeatherFrame:
    return WeatherFrame(
        airports=frame.airports[mask],
        time=frame.time[mask],
        wind_speed=frame.wind_speed[mask],
        wind_dir=frame.wind_dir[mask],
        temp=frame.temp[mask],
        vis=frame.vis[mask],
        hour=frame.hour[mask],
        runway_heading=frame.runway_heading[mask],
        event_id=frame.event_id[mask],
        y=frame.y[mask],
        hazard=frame.hazard[mask],
        hazard_threshold=frame.hazard_threshold,
        source=frame.source,
    )


def _simulate_airport(
    rng: np.random.Generator,
    airport: str,
    hours: np.ndarray,
    *,
    climate: str,
    storm: np.ndarray,
) -> WeatherFrame:
    n = int(hours.shape[0])
    runway = {"ZBAA": 36.0, "ZSPD": 16.0, "ZGGG": 3.0, "ZUUU": 2.0, "ZBHH": 26.0, "ZJHK": 9.0}.get(airport, 18.0)
    diurnal = 4.0 * np.sin(2 * np.pi * (hours % 24) / 24.0)
    base = {"temperate": 6.0, "plateau": 8.5, "tropical": 5.0}[climate]
    wind_speed = np.clip(base + diurnal + rng.normal(0, 1.4, n), 0.3, 28.0)
    wind_speed = np.where(storm, wind_speed + 9.0, wind_speed)
    wind_dir = (rng.normal(220.0 if climate != "tropical" else 90.0, 35.0, n)) % 360.0
    temp = {"temperate": 12.0, "plateau": 6.0, "tropical": 26.0}[climate] + rng.normal(0, 3.0, n)
    vis = np.clip(8000 - 80 * np.maximum(wind_speed - 8, 0) + rng.normal(0, 200, n), 200, 10000)
    hour = hours % 24
    rad = np.deg2rad(wind_dir - runway)
    headwind = wind_speed * np.cos(rad)
    coef = {"temperate": (0.55, 0.28, 0.08), "plateau": (0.40, 0.15, 0.20), "tropical": (0.70, 0.05, 0.02)}[climate]
    noise = rng.normal(0, 0.55, n)
    y = coef[0] * wind_speed + coef[1] * headwind + coef[2] * (temp / 10.0) + noise
    y = np.clip(y, 0.0, 32.0)
    event_id = np.where(storm, np.where(hours < 200, "storm_a", "storm_b"), "none")
    return WeatherFrame(
        airports=np.full(n, airport),
        time=hours.astype(int),
        wind_speed=wind_speed,
        wind_dir=wind_dir,
        temp=temp,
        vis=vis,
        hour=hour.astype(float),
        runway_heading=np.full(n, runway),
        event_id=event_id,
        y=y,
        hazard=y >= HAZARD_THRESHOLD,
        source=airport,
    )
