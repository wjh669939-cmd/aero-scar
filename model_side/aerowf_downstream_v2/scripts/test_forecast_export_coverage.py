#!/usr/bin/env python3
"""CPU check: train/val still drop incomplete targets; export covers 0..3960 × 3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path("/root/autodl-tmp/aerowf_downstream_v2/src")
sys.path.insert(0, str(SRC))

from aerowf_forecast_train_v2 import (  # noqa: E402
    AIRPORTS,
    AirportForecastDataset,
    CONTRACT_VAL_ROWS,
    CONTRACT_VAL_ROWS_PER_AIRPORT,
    assert_full_val_export_coverage,
)

VAL_ROOT = Path(
    "/root/autodl-tmp/aerowf_delivery/v1/extracted/"
    "AeroWF_v1_MODEL_TRAINING/release_v1/trainval/val"
)


def main() -> None:
    train_like = []
    export_like = []
    for airport in AIRPORTS:
        complete = AirportForecastDataset(VAL_ROOT, airport)
        full = AirportForecastDataset(
            VAL_ROOT, airport, require_complete_horizons=False
        )
        train_like.append(complete)
        export_like.append(full)
        n_file = int(np.load(VAL_ROOT / airport / "source_index.npy").shape[0])
        assert n_file == CONTRACT_VAL_ROWS_PER_AIRPORT, (airport, n_file)
        assert len(complete) == n_file - 8, (airport, len(complete), n_file)
        assert len(full) == n_file, (airport, len(full), n_file)
        src = np.asarray(full.source_index[full.anchors], dtype=np.int64)
        assert set(src.tolist()) == set(range(n_file)), airport
        tail = full[n_file - 1]
        assert int(tail["source_index"]) == n_file - 1
        assert np.isfinite(np.asarray(full[0]["target"])).all()
        # last row has no T+8 partner; prediction path still returns an input window
        assert tail["x"].shape[0] > 0
        assert np.isnan(np.asarray(tail["target"])).any()

    fake = {
        "prediction": np.zeros((CONTRACT_VAL_ROWS, 4, 3, 2), dtype=np.float32),
        "airport_id": np.repeat(np.arange(3), CONTRACT_VAL_ROWS_PER_AIRPORT),
        "source_index": np.tile(np.arange(CONTRACT_VAL_ROWS_PER_AIRPORT), 3),
    }
    assert_full_val_export_coverage(fake)
    print(
        "ok",
        "complete_val",
        sum(len(part) for part in train_like),
        "export_val",
        sum(len(part) for part in export_like),
    )


if __name__ == "__main__":
    main()
