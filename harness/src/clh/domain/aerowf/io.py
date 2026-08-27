"""Index.csv + frozen-stats IO. Does not refit normalisation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from clh.core.errors import EvaluatorError
from clh.domain.aerowf.contract import (
    FORBIDDEN_SEARCH_TOKENS,
    RUNWAY_FEATURE_NAMES,
)


@dataclass
class IndexRow:
    sample_id: str
    airport: str
    timestamp: np.datetime64
    source_split: str
    role: str
    source_index: int


def assert_search_path_allowed(path: Path) -> None:
    text = str(path).replace("\\", "/")
    for token in FORBIDDEN_SEARCH_TOKENS:
        needle = token.replace("\\", "/")
        if needle in text:
            raise EvaluatorError(
                f"search/training path is forbidden by MODEL_HANDOFF_v1.md: {path} (matched {token})"
            )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_stats(release_root: Path, name: str) -> dict[str, Any]:
    path = release_root / "metadata" / name
    if not path.is_file():
        raise EvaluatorError(f"missing frozen stats {path}")
    return load_json(path)


def runway_minmax(stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features = stats["runway"]["features"]
    mins = np.zeros((len(RUNWAY_FEATURE_NAMES),), dtype=np.float32)
    denoms = np.ones((len(RUNWAY_FEATURE_NAMES),), dtype=np.float32)
    if isinstance(features, dict):
        iterable = [features[name] for name in RUNWAY_FEATURE_NAMES]
    else:
        iterable = sorted(features, key=lambda row: int(row["index"]))
    for i, block in enumerate(iterable):
        mins[i] = float(block["min"])
        denoms[i] = float(block["denominator"]) if float(block["denominator"]) else 1.0
    return mins, denoms


def apply_runway_minmax(raw: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mins, denoms = runway_minmax(stats)
    scaled = (raw - mins.reshape(1, 1, 1, -1)) / denoms.reshape(1, 1, 1, -1)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def invert_runway_minmax(scaled: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mins, denoms = runway_minmax(stats)
    return (scaled * denoms.reshape(1, 1, 1, -1) + mins.reshape(1, 1, 1, -1)).astype(np.float32)


def read_index_csv(path: Path) -> list[IndexRow]:
    rows: list[IndexRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            stamp = _parse_timestamp(raw["timestamp"])
            rows.append(
                IndexRow(
                    sample_id=str(raw["sample_id"]),
                    airport=str(raw["airport"]),
                    timestamp=stamp,
                    source_split=str(raw.get("source_split") or raw.get("split") or ""),
                    role=str(raw.get("role") or ""),
                    source_index=int(raw.get("source_index") or raw.get("local_index") or 0),
                )
            )
    return rows


def subsample_rows(rows: list[IndexRow], max_samples: int, seed: int) -> list[IndexRow]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = np.random.default_rng(seed)
    # Even time coverage: take equally spaced indices, then jitter none — deterministic.
    positions = np.linspace(0, len(rows) - 1, max_samples)
    chosen = np.unique(np.rint(positions).astype(int))
    if chosen.size < max_samples:
        extra = rng.choice(len(rows), size=max_samples - chosen.size, replace=False)
        chosen = np.unique(np.concatenate([chosen, extra]))[:max_samples]
    return [rows[i] for i in chosen.tolist()]


def split_dir(release_root: Path, family: str, split: str, airport: str) -> Path:
    return release_root / family / split / airport


def tensors_present(folder: Path) -> bool:
    return (folder / "runway.npy").is_file() and (folder / "runway_mask.npy").is_file()


def load_tensor_folder(folder: Path) -> dict[str, np.ndarray]:
    missing = [name for name in ("runway.npy", "runway_mask.npy", "exo_continuous.npy") if not (folder / name).is_file()]
    if missing:
        raise EvaluatorError(f"{folder} missing tensors {missing}")
    payload = {
        "runway": np.load(folder / "runway.npy"),
        "runway_mask": np.load(folder / "runway_mask.npy"),
        "exo_continuous": np.load(folder / "exo_continuous.npy"),
    }
    label_path = folder / "weather_label.npy"
    if label_path.is_file():
        payload["weather_label"] = np.load(label_path)
    cat_path = folder / "exo_categorical.npy"
    if cat_path.is_file():
        payload["exo_categorical"] = np.load(cat_path)
    return payload


def _parse_timestamp(text: str) -> np.datetime64:
    raw = text.strip().replace("Z", "")
    if raw.endswith("000000000"):
        raw = raw[:19]
    try:
        return np.datetime64(raw)
    except ValueError:
        dt = datetime.fromisoformat(raw[:19])
        return np.datetime64(dt.isoformat())
