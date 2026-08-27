"""Domain adapter seam. Dummy synthetic weather and AeroWF share this contract."""

from __future__ import annotations

from typing import Any, Protocol

from clh.config import HarnessConfig
from clh.domain.atc.leakage import filter_external_source as dummy_filter
from clh.domain.dummy.weather import (
    SyntheticAirportWeather,
    WeatherFrame,
    build_weather,
    concat_frames,
    load_split,
)
from clh.domain.metrics import score_predictions
from clh.research.cards import MetricsBundle


class DomainAdapter(Protocol):
    name: str

    def describe(self) -> str: ...

    def load_split(self, split: str) -> Any: ...

    def extra_frame(self, source_id: str) -> Any: ...

    def filter_external(self, source_id: str, frame: Any) -> tuple[Any, Any]: ...

    def concat(self, frames: list[Any]) -> Any: ...

    def score(self, y_hat, frame, *, split: str) -> MetricsBundle: ...


class DummyATCAdapter:
    name = "dummy"

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self._weather = build_weather(
            source_airports=tuple(config.domain.source_airports),
            holdout_airports=tuple(config.domain.holdout_airports),
        )

    def weather(self) -> SyntheticAirportWeather:
        return self._weather

    def describe(self) -> str:
        return (
            "Dummy aerodrome-weather domain for Closed-loop Auto Research. "
            "Axes: representation, model, physics, data. "
            "Hidden tests: temporal, spatial, extreme-event."
        )

    def load_split(self, split: str) -> WeatherFrame:
        return load_split(self._weather, split)  # type: ignore[arg-type]

    def extra_frame(self, source_id: str) -> WeatherFrame:
        if source_id not in self._weather.frames:
            raise KeyError(source_id)
        return self._weather.frames[source_id]

    def filter_external(self, source_id: str, frame: WeatherFrame):
        return dummy_filter(self._weather, source_id, frame)

    def concat(self, frames: list[WeatherFrame]) -> WeatherFrame:
        return concat_frames(frames)

    def score(self, y_hat, frame, *, split: str) -> MetricsBundle:
        return score_predictions(y_hat, frame, split=split)


class AeroWFAdapter:
    name = "aerowf"

    def __init__(self, config: HarnessConfig, workspace) -> None:
        from clh.domain.aerowf.world import build_aerowf_world

        self.config = config
        self.world = build_aerowf_world(config, workspace)

    def describe(self) -> str:
        notes = "; ".join(self.world.notes) if self.world.notes else "official splits"
        return (
            "AeroWF Data Contract V1 closed-loop domain. "
            "Search-visible: release_v1/trainval/{train,val} (and pretrain as data-axis extras). "
            "Forbidden: sealed/**, ZBAD, pretrain/test. "
            f"Airports={list(self.world.source_airports)}. {notes}."
        )

    def load_split(self, split: str):
        return self.world.load_split(split)

    def extra_frame(self, source_id: str):
        return self.world.extra_frame(source_id)

    def filter_external(self, source_id: str, frame):
        return self.world.filter_external(source_id, frame)

    def concat(self, frames: list):
        return self.world.concat(frames)

    def score(self, y_hat, frame, *, split: str) -> MetricsBundle:
        return score_predictions(y_hat, frame, split=split)


def load_adapter(config: HarnessConfig, workspace=None) -> DummyATCAdapter | AeroWFAdapter:
    name = (config.domain.name or "dummy").lower()
    if name in {"aerowf", "atc"}:
        from pathlib import Path

        if workspace is None:
            workspace = Path.cwd()
        return AeroWFAdapter(config, Path(workspace))
    return DummyATCAdapter(config)
