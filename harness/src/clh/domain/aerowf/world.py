"""Assemble the search-visible AeroWF world and evaluator-owned held-out splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clh.config import HarnessConfig
from clh.core.errors import EvaluatorError
from clh.domain.aerowf.contract import (
    DOWNSTREAM_STATS_NAME,
    PRETRAIN_STATS_NAME,
    SPATIAL_HOLDOUT_AIRPORT,
    TRAINING_AIRPORTS,
)
from clh.domain.aerowf.frames import AeroFrame, concat_frames, last_step_wind_speed, mask_frame
from clh.domain.aerowf.io import (
    IndexRow,
    assert_search_path_allowed,
    invert_runway_minmax,
    load_frozen_stats,
    load_tensor_folder,
    read_index_csv,
    split_dir,
    subsample_rows,
    tensors_present,
)
from clh.domain.aerowf.leakage import LeakageDecision, filter_external_source
from clh.domain.aerowf.materialize import materialize_frame, rows_from_stamps


@dataclass
class AeroWorld:
    release_root: Path
    stats: dict
    frames: dict[str, AeroFrame]
    sealed_available: bool
    source_airports: tuple[str, ...]
    notes: list[str] = field(default_factory=list)

    def load_split(self, split: str) -> AeroFrame:
        if split not in self.frames:
            raise EvaluatorError(f"unknown split {split}")
        return self.frames[split]

    def extra_frame(self, source_id: str) -> AeroFrame:
        if source_id not in self.frames:
            raise EvaluatorError(f"unknown extra source {source_id}")
        return self.frames[source_id]

    def filter_external(self, source_id: str, frame: AeroFrame) -> tuple[AeroFrame | None, LeakageDecision]:
        return filter_external_source(self.frames, source_id, frame)

    def concat(self, frames: list[AeroFrame]) -> AeroFrame:
        return concat_frames(frames)


def build_aerowf_world(config: HarnessConfig, workspace: Path) -> AeroWorld:
    release = config.resolved_data_root(workspace)
    if release is None or not release.is_dir():
        raise EvaluatorError(
            "AeroWF domain requires domain.data_root pointing at "
            "AeroWF_v1_MODEL_TRAINING/release_v1"
        )
    airports = tuple(config.domain.source_airports) or TRAINING_AIRPORTS
    stats = load_frozen_stats(release, DOWNSTREAM_STATS_NAME)
    max_n = config.domain.max_samples_per_split
    seed = config.domain.inner_split_seed
    notes: list[str] = []

    train = _load_family(
        release, stats, "trainval", "train", airports, max_n=max_n, seed=seed, search_visible=True
    )
    val_full = _load_family(
        release, stats, "trainval", "val", airports, max_n=max_n, seed=seed + 1, search_visible=True
    )

    sealed_root = config.resolved_sealed_root(workspace)
    if sealed_root is None:
        candidate = release / "sealed"
        sealed_root = candidate if candidate.is_dir() else None
    sealed_available = bool(sealed_root and sealed_root.is_dir())

    if sealed_available:
        val = val_full
        test_temporal = _try_load_sealed(sealed_root, stats, "temporal", airports, max_n, seed + 2)
        test_spatial = _try_load_sealed(
            sealed_root, stats, "spatial", (SPATIAL_HOLDOUT_AIRPORT,), max_n, seed + 3
        )
        notes.append("using official sealed evaluation partitions")
    else:
        val, test_temporal = _carve_temporal_holdout(val_full, frac=config.domain.inner_val_frac)
        test_spatial = _synthetic_spatial(val, stats, seed=seed + 11)
        notes.append(
            "sealed pack not mounted; temporal holdout is the last "
            f"{config.domain.inner_val_frac:.0%} of trainval/val by timestamp; "
            "spatial is a contract-climate ZBAD probe (not the official sealed/spatial pack)"
        )

    if test_temporal is None or len(test_temporal) == 0:
        raise EvaluatorError("temporal held-out split is empty")
    if test_spatial is None or len(test_spatial) == 0:
        test_spatial = _synthetic_spatial(test_temporal, stats, seed=seed + 11)

    event_mask = test_temporal.hazard | (test_temporal.event_id == "storm")
    if int(event_mask.sum()) < 8:
        order = np.argsort(-test_temporal.y)
        event_mask = np.zeros(len(test_temporal), dtype=bool)
        event_mask[order[: max(8, len(test_temporal) // 5)]] = True
    test_event = mask_frame(test_temporal, event_mask)
    test_event.source = "test_event"

    extras = _build_extras(release, stats, train, val, test_temporal, airports, max_n, seed)
    frames = {
        "train": train,
        "val": val,
        "test_temporal": test_temporal,
        "test_spatial": test_spatial,
        "test_event": test_event,
        **extras,
    }
    return AeroWorld(
        release_root=release,
        stats=stats,
        frames=frames,
        sealed_available=sealed_available,
        source_airports=airports,
        notes=notes,
    )


def _load_family(
    release: Path,
    stats: dict,
    family: str,
    split: str,
    airports: tuple[str, ...],
    *,
    max_n: int,
    seed: int,
    search_visible: bool,
) -> AeroFrame:
    parts: list[AeroFrame] = []
    for i, airport in enumerate(airports):
        folder = split_dir(release, family, split, airport)
        if search_visible:
            assert_search_path_allowed(folder)
        if not folder.is_dir():
            raise EvaluatorError(f"missing AeroWF split folder {folder}")
        index_path = folder / "index.csv"
        if not index_path.is_file():
            raise EvaluatorError(f"missing index.csv in {folder}")
        rows = subsample_rows(read_index_csv(index_path), max_n, seed + i)
        if tensors_present(folder):
            parts.append(_frame_from_tensors(folder, rows, stats, source=f"{family}/{split}", role=f"{family}_{split}"))
        else:
            parts.append(
                materialize_frame(
                    rows,
                    stats,
                    source=f"{family}/{split}",
                    role=f"{family}_{split}",
                    seed=seed + i,
                )
            )
    return concat_frames(parts)


def _frame_from_tensors(
    folder: Path,
    rows: list[IndexRow],
    stats: dict,
    *,
    source: str,
    role: str,
) -> AeroFrame:
    tensors = load_tensor_folder(folder)
    n = tensors["runway"].shape[0]
    idx = np.array([min(row.source_index, n - 1) for row in rows], dtype=int)
    runway = tensors["runway"][idx]
    mask = tensors["runway_mask"][idx].astype(bool)
    exo = tensors["exo_continuous"][idx]
    physical = invert_runway_minmax(runway, stats)
    prev = last_step_wind_speed(physical, mask, step=-2)
    weather_label = tensors["weather_label"][idx] if "weather_label" in tensors else None
    stamps = np.array([row.timestamp for row in rows], dtype="datetime64[s]")
    hours = ((stamps.astype("datetime64[m]").astype(np.int64) % (24 * 60)) / 60.0).astype(np.float32)
    y = (0.70 * prev + 0.11 * hours + 0.45).astype(np.float32)
    return AeroFrame(
        airports=np.array([row.airport for row in rows], dtype=object),
        time=stamps,
        sample_id=np.array([row.sample_id for row in rows], dtype=object),
        runway=runway.astype(np.float32),
        runway_mask=mask,
        exo_continuous=exo.astype(np.float32),
        y=y,
        hazard=y >= 12.0,
        wind_speed=y,
        prev_wind_speed=prev.astype(np.float32),
        hour=hours,
        event_id=np.array(["none"] * len(rows), dtype=object),
        weather_label=weather_label,
        exo_categorical=tensors["exo_categorical"][idx] if "exo_categorical" in tensors else None,
        source=source,
        role=role,
        stats_name=DOWNSTREAM_STATS_NAME,
    )


def _carve_temporal_holdout(val: AeroFrame, *, frac: float) -> tuple[AeroFrame, AeroFrame]:
    keep_val = np.zeros(len(val), dtype=bool)
    keep_test = np.zeros(len(val), dtype=bool)
    for airport in sorted(set(map(str, val.airports))):
        idx = np.where(val.airports == airport)[0]
        order = idx[np.argsort(val.time[idx])]
        cut = int(round((1.0 - frac) * len(order)))
        cut = min(max(1, cut), len(order) - 1) if len(order) >= 2 else 0
        keep_val[order[:cut]] = True
        keep_test[order[cut:]] = True
    search_val = mask_frame(val, keep_val)
    search_val.source = "trainval/val"
    temporal = mask_frame(val, keep_test)
    temporal.source = "holdout_temporal"
    return search_val, temporal


def _synthetic_spatial(template: AeroFrame, stats: dict, *, seed: int) -> AeroFrame:
    stamps = template.time[: max(16, min(len(template), 256))]
    rows = rows_from_stamps(SPATIAL_HOLDOUT_AIRPORT, stamps, prefix="probe")
    frame = materialize_frame(
        rows,
        stats,
        source="synthetic_spatial_probe",
        role="spatial_probe",
        climate_override=SPATIAL_HOLDOUT_AIRPORT,
        seed=seed,
    )
    return frame


def _try_load_sealed(
    sealed_root: Path,
    stats: dict,
    kind: str,
    airports: tuple[str, ...],
    max_n: int,
    seed: int,
) -> AeroFrame | None:
    folder = sealed_root / kind
    if kind == "spatial":
        folder = sealed_root / "spatial" / SPATIAL_HOLDOUT_AIRPORT
        if folder.is_dir() and (folder / "index.csv").is_file():
            return _load_one_folder(folder, stats, max_n, seed, source=f"sealed/{kind}", role=f"sealed_{kind}")
        parent = sealed_root / "spatial"
        if parent.is_dir():
            parts = []
            for child in sorted(parent.iterdir()):
                if child.is_dir() and (child / "index.csv").is_file():
                    parts.append(
                        _load_one_folder(child, stats, max_n, seed, source="sealed/spatial", role="sealed_spatial")
                    )
            return concat_frames(parts) if parts else None
        return None
    if not folder.is_dir():
        return None
    parts = []
    for airport in airports:
        child = folder / airport
        if child.is_dir() and (child / "index.csv").is_file():
            parts.append(_load_one_folder(child, stats, max_n, seed, source=f"sealed/{kind}", role=f"sealed_{kind}"))
    return concat_frames(parts) if parts else None


def _load_one_folder(folder: Path, stats: dict, max_n: int, seed: int, *, source: str, role: str) -> AeroFrame:
    rows = subsample_rows(read_index_csv(folder / "index.csv"), max_n, seed)
    if tensors_present(folder):
        return _frame_from_tensors(folder, rows, stats, source=source, role=role)
    return materialize_frame(rows, stats, source=source, role=role, seed=seed)


def _build_extras(
    release: Path,
    stats: dict,
    train: AeroFrame,
    val: AeroFrame,
    temporal: AeroFrame,
    airports: tuple[str, ...],
    max_n: int,
    seed: int,
) -> dict[str, AeroFrame]:
    extras: dict[str, AeroFrame] = {}
    pretrain_dir = release / "pretrain" / "train"
    if pretrain_dir.is_dir():
        try:
            pretrain_stats = load_frozen_stats(release, PRETRAIN_STATS_NAME)
            extras["pretrain_train"] = _load_family(
                release,
                pretrain_stats,
                "pretrain",
                "train",
                airports,
                max_n=max_n,
                seed=seed + 21,
                search_visible=True,
            )
            extras["pretrain_train"].source = "pretrain_train"
            extras["pretrain_train"].stats_name = PRETRAIN_STATS_NAME
        except EvaluatorError:
            pass
    extras["matched_climate"] = extras.get("pretrain_train", train)
    leak_src = concat_frames([val, temporal])
    leak_src.source = "leak_val"
    extras["leak_val"] = leak_src
    extras["same_source_leak"] = leak_src
    shifted_rows = rows_from_stamps("ZJHK", train.time[: min(len(train), max(32, max_n or 64))], prefix="shifted")
    extras["shifted_climate"] = materialize_frame(
        shifted_rows, stats, source="shifted_climate", role="external", climate_override="ZJHK", seed=seed + 33
    )
    extras["shifted_ZJHK"] = extras["shifted_climate"]
    extras["matched_ZBHH"] = extras["matched_climate"]
    return extras
