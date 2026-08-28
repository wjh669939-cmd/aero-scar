"""Persistence reference baseline for AeroWF forecast (D3 收尾).

预测规则：T+1/T+4/T+8 的预测 = anchor 时刻（输入窗最后一帧，internal index 95）
的 (wind_x, wind_y) 观测值。全覆盖导出（11883 行），与正式管线同一 npz 契约，
经 predictions_adapter + C1 evaluator 同一口径评测。

只读冻结数据；不训练、不碰 GPU。
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

AIRPORTS = ("ZBAA", "ZSPD", "ZSSS")
AIRPORT_TO_ID = {name: index for index, name in enumerate(AIRPORTS)}
HORIZONS = OrderedDict((("T+1", 15), ("T+4", 60), ("T+8", 120)))
TARGET_INTERNAL_INDEX = 95
NS_PER_MINUTE = 60 * 1_000_000_000
CONTRACT_VAL_ROWS_PER_AIRPORT = 3961

DATA_ROOT = Path(
    "/root/autodl-tmp/aerowf_delivery/v1/extracted/AeroWF_v1_MODEL_TRAINING/release_v1/trainval"
)
OUT_DIR = Path("/root/autodl-tmp/aerowf_downstream_v2/results/references/persistence_forecast")


def export_airport(split_root: Path, airport: str) -> dict[str, np.ndarray]:
    root = split_root / airport
    runway = np.load(root / "runway.npy", mmap_mode="r")
    runway_mask = np.asarray(np.load(root / "runway_mask.npy"), dtype=bool)
    timestamps = np.asarray(np.load(root / "timestamps.npy"))
    source_index = np.asarray(np.load(root / "source_index.npy"), dtype=np.int64)

    timestamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    if len(np.unique(timestamp_ns)) != len(timestamp_ns):
        raise RuntimeError(f"{root}: duplicate timestamps")
    lookup = {int(ts): idx for idx, ts in enumerate(timestamp_ns)}

    rows = runway.shape[0]
    n_slots = runway.shape[1]
    current = np.asarray(runway[:, :, TARGET_INTERNAL_INDEX, 1:3], dtype=np.float32)

    prediction = np.repeat(current[:, :, None, :], len(HORIZONS), axis=2)
    target = np.full((rows, n_slots, len(HORIZONS), 2), np.nan, dtype=np.float32)
    for row in range(rows):
        for h_idx, minutes in enumerate(HORIZONS.values()):
            partner = lookup.get(int(timestamp_ns[row] + minutes * NS_PER_MINUTE), -1)
            if partner >= 0:
                target[row, :, h_idx, :] = current[partner]

    # 虚拟槽位从未被训练/计分，中和为 0.5（与 adapter 的处理语义一致），
    # 保证全阵列有限性断言可过。
    virtual = ~runway_mask
    if virtual.any():
        prediction[virtual, :, :] = 0.5

    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{airport}: non-finite persistence prediction on real slots")

    return {
        "prediction": prediction,
        "target": target,
        "node_mask": runway_mask,
        "airport_id": np.full(rows, AIRPORT_TO_ID[airport], dtype=np.int64),
        "anchor_timestamp_ns": timestamp_ns,
        "source_index": source_index,
    }


def main() -> None:
    val_root = DATA_ROOT / "val"
    collected: dict[str, list[np.ndarray]] = {}
    for airport in AIRPORTS:
        arrays = export_airport(val_root, airport)
        n = arrays["source_index"].shape[0]
        print(f"{airport}: rows={n}")
        for key, value in arrays.items():
            collected.setdefault(key, []).append(value)
    merged = {key: np.concatenate(values) for key, values in collected.items()}

    rows = merged["source_index"].shape[0]
    if rows != CONTRACT_VAL_ROWS_PER_AIRPORT * len(AIRPORTS):
        raise RuntimeError(f"rows {rows} != contract 11883")
    for airport_id, airport in enumerate(AIRPORTS):
        have = set(merged["source_index"][merged["airport_id"] == airport_id].tolist())
        if have != set(range(CONTRACT_VAL_ROWS_PER_AIRPORT)):
            raise RuntimeError(f"{airport}: source_index coverage incomplete")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "validation_predictions.npz"
    np.savez_compressed(out, **merged)
    print("saved:", out, "rows:", rows)


if __name__ == "__main__":
    main()
