"""Evaluator-owned leakage filter (paper §3.7 mapped in docs/axes/04-axis-data.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from clh.domain.aerowf.contract import NEAR_ANALOGUE_MINUTES, SAME_SOURCE_RATE
from clh.domain.aerowf.frames import AeroFrame, mask_frame


@dataclass
class LeakageDecision:
    source_id: str
    admitted: bool
    reason: str
    overlap_rate: float = 0.0
    removed_rows: int = 0
    kept_rows: int = 0
    layers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "admitted": self.admitted,
            "reason": self.reason,
            "overlap_rate": self.overlap_rate,
            "removed_rows": self.removed_rows,
            "kept_rows": self.kept_rows,
            "layers": list(self.layers),
        }


def hidden_identity_sets(frames: dict[str, AeroFrame]) -> dict[str, set[tuple[str, str]]]:
    names = [name for name in frames if name.startswith("val") or name.startswith("test")]
    return {name: frames[name].identities() for name in names}


def filter_external_source(
    frames: dict[str, AeroFrame],
    source_id: str,
    extra: AeroFrame,
) -> tuple[AeroFrame | None, LeakageDecision]:
    """L1 identity, L2 same-source >5% of hidden keys, L3 near-analogue, L4 causal."""
    hidden_ids: set[tuple[str, str]] = set()
    hidden_times: set[tuple[str, np.datetime64]] = set()
    test_ids: set[tuple[str, str]] = set()
    for name, frame in frames.items():
        if name.startswith("test") or name == "val":
            hidden_ids |= frame.identities()
            hidden_times |= frame.time_identities()
        if name.startswith("test"):
            test_ids |= frame.identities()
    train = frames["train"]
    extra_ids = extra.identities()
    denom = max(1, len(test_ids) if test_ids else len(hidden_ids))
    overlap_rate = len(extra_ids & test_ids) / denom
    if overlap_rate > SAME_SOURCE_RATE:
        return None, LeakageDecision(
            source_id=source_id,
            admitted=False,
            reason="same-source rejection: hidden-split overlap exceeds 5%",
            overlap_rate=overlap_rate,
            removed_rows=len(extra),
            kept_rows=0,
            layers=["identity", "same_source"],
        )

    keep = np.ones(len(extra), dtype=bool)
    extra_times = extra.time.astype("datetime64[s]")
    train_times = train.time.astype("datetime64[s]")
    max_train = train_times.max() if len(train) else np.datetime64("1970-01-01", "s")
    # L4 causal: extra evidence cannot be from after the training horizon.
    keep &= extra_times <= max_train
    # L1 identity against train/val/test sample_id
    blocked = hidden_ids | train.identities()
    for i, ident in enumerate(zip(extra.airports, extra.sample_id)):
        if (str(ident[0]), str(ident[1])) in blocked:
            keep[i] = False
    # L3 near-analogue: same airport and |Δt| within lookback+horizon of a hidden sample
    if hidden_times:
        hidden_by_airport: dict[str, list] = {}
        for airport, stamp in hidden_times:
            hidden_by_airport.setdefault(airport, []).append(stamp)
        hidden_arr = {k: np.array(v, dtype="datetime64[s]") for k, v in hidden_by_airport.items()}
        window = np.timedelta64(NEAR_ANALOGUE_MINUTES, "m")
        for i in np.where(keep)[0]:
            airport = str(extra.airports[i])
            if airport not in hidden_arr:
                continue
            delta = np.abs(hidden_arr[airport] - extra_times[i])
            if np.any(delta <= window):
                keep[i] = False

    kept = int(keep.sum())
    if kept == 0:
        return None, LeakageDecision(
            source_id=source_id,
            admitted=False,
            reason="all rows removed by identity/causal/near-analogue filters",
            overlap_rate=overlap_rate,
            removed_rows=len(extra),
            kept_rows=0,
            layers=["identity", "causal", "near_analogue"],
        )
    admitted = mask_frame(extra, keep)
    admitted.source = source_id
    return admitted, LeakageDecision(
        source_id=source_id,
        admitted=True,
        reason="admitted after identity, same-source, near-analogue, and causal filters",
        overlap_rate=overlap_rate,
        removed_rows=len(extra) - kept,
        kept_rows=kept,
        layers=["identity", "same_source", "near_analogue", "causal"],
    )
