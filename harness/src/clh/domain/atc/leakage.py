"""Meteorological leakage filter. Evaluator-owned; agents cannot bypass it."""

from __future__ import annotations

from dataclasses import dataclass, field

from clh.domain.dummy.weather import WeatherFrame, hidden_identity_sets, SyntheticAirportWeather

SAME_SOURCE_RATE = 0.05


@dataclass
class LeakageDecision:
    source_id: str
    admitted: bool
    reason: str
    overlap_rate: float = 0.0
    removed_rows: int = 0
    kept_rows: int = 0
    layers: list[str] = field(default_factory=list)


def filter_external_source(
    weather: SyntheticAirportWeather,
    source_id: str,
    frame: WeatherFrame,
) -> tuple[WeatherFrame | None, LeakageDecision]:
    hidden = hidden_identity_sets(weather)
    test_ids = set()
    for key in ("test_temporal", "test_spatial", "test_event", "val"):
        test_ids |= hidden[key]
    identities = frame.identities()
    overlap = identities & test_ids
    overlap_rate = len(overlap) / max(1, len(hidden["test_temporal"] | hidden["test_spatial"] | hidden["test_event"]))
    if overlap_rate > SAME_SOURCE_RATE:
        return None, LeakageDecision(
            source_id=source_id,
            admitted=False,
            reason="same-source rejection: hidden-split overlap exceeds 5%",
            overlap_rate=overlap_rate,
            removed_rows=len(frame),
            kept_rows=0,
            layers=["identity", "same_source"],
        )
    train = weather.frames["train"]
    max_train_t = int(train.time.max())
    causal_mask = frame.time <= max_train_t
    event_blocked = frame.event_id != "none"
    keep = causal_mask & ~event_blocked
    for i, ident in enumerate(zip(frame.airports, frame.time)):
        if (str(ident[0]), int(ident[1])) in (train.identities() | test_ids):
            keep[i] = False
    kept = int(keep.sum())
    if kept == 0:
        return None, LeakageDecision(
            source_id=source_id,
            admitted=False,
            reason="all rows removed by identity/causal/event filters",
            overlap_rate=overlap_rate,
            removed_rows=len(frame),
            kept_rows=0,
            layers=["identity", "causal", "event"],
        )
    admitted = _mask_frame(frame, keep)
    admitted.source = source_id
    return admitted, LeakageDecision(
        source_id=source_id,
        admitted=True,
        reason="admitted after identity, causal, and event filters",
        overlap_rate=overlap_rate,
        removed_rows=len(frame) - kept,
        kept_rows=kept,
        layers=["identity", "causal", "event"],
    )


def _mask_frame(frame: WeatherFrame, keep) -> WeatherFrame:
    from clh.domain.dummy.weather import WeatherFrame as Frame

    return Frame(
        airports=frame.airports[keep],
        time=frame.time[keep],
        wind_speed=frame.wind_speed[keep],
        wind_dir=frame.wind_dir[keep],
        temp=frame.temp[keep],
        vis=frame.vis[keep],
        hour=frame.hour[keep],
        runway_heading=frame.runway_heading[keep],
        event_id=frame.event_id[keep],
        y=frame.y[keep],
        hazard=frame.hazard[keep],
        hazard_threshold=frame.hazard_threshold,
        source=frame.source,
    )
