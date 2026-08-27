"""G-10：D 阶段产物 validation_predictions.npz → C 评测器合同格式适配器。

合同（C1 evaluator v1.0）：
- npz 恰有 {sample_id, pred} 两键；
- sample_id = "processed:{airport}:val:{index:08d}"，必须覆盖全量 val 清单；
- forecast pred 尾形状 (4,3,2) float、值域 [0,1]；classification pred 标量 int ∈ {0,1,2}。

已知接口缺口（8/27 发现）：D 的 forecast 数据集因训练目标完整性丢弃每机场尾部
8 条样本（共 24 条），但评测器要求全覆盖且其中部分行会在 T+1/T+4 被计分。
因此本适配器默认 missing 即报错；只有显式 allow_fill=True（仅限接口联调，
trial_id 必须带 interface_test 标记）才允许填充占位值。正式 trial 必须由 D 侧
全覆盖导出修复（见 19 文档接口缺口通知）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

AIRPORT_ORDER = ("ZBAA", "ZSPD", "ZSSS")  # 与 D 侧 AIRPORTS 及评测器 config 一致
FORECAST_TAIL_SHAPE = (4, 3, 2)
FILL_VALUE_FORECAST = 0.5  # 值域中点；仅接口测试可用


class AdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdaptResult:
    out_path: Path
    n_rows: int
    n_missing_filled: int
    missing_sample_ids: tuple[str, ...]


def _sample_id(airport: str, index: int) -> str:
    return f"processed:{airport}:val:{index:08d}"


def full_manifest_ids(val_counts: dict[str, int]) -> list[str]:
    ids: list[str] = []
    for airport in AIRPORT_ORDER:
        ids.extend(_sample_id(airport, i) for i in range(val_counts[airport]))
    return ids


def adapt_stage_npz(
    stage_npz: Path,
    task: str,
    out_path: Path,
    val_counts: dict[str, int],
    allow_fill: bool = False,
) -> AdaptResult:
    """转换一个阶段的 validation_predictions.npz 为合同 npz。

    val_counts: 每机场全量 val 样本数（以 C 清单为准，当前 {各 3961}）。
    """
    if task not in ("forecast", "classification"):
        raise AdapterError(f"unsupported task: {task}")

    with np.load(stage_npz, allow_pickle=False) as z:
        prediction = np.asarray(z["prediction"])
        airport_id = np.asarray(z["airport_id"])
        source_index = np.asarray(z["source_index"])
        node_mask = np.asarray(z["node_mask"]) if "node_mask" in z.files else None

    # 虚拟跑道槽位中和（仅 forecast）：padding 槽位从未被训练、评测器按
    # runway_mask 永不计分，但值域检查发生在掩码之前——原样提交必被拒。
    # 置为域内常数不属于 clip 作弊（G-10 联调实测：seed43 收敛模型越界值
    # 99.99% 在虚拟槽位）。真实跑道的越界绝不修改，照实提交由评测器裁决。
    if task == "forecast" and node_mask is not None:
        virtual = ~node_mask.astype(bool)  # (N, 4)
        prediction = prediction.copy()
        prediction[virtual] = FILL_VALUE_FORECAST

    have: dict[str, np.ndarray] = {}
    for aid, airport in enumerate(AIRPORT_ORDER):
        rows = airport_id == aid
        idx = source_index[rows]
        preds = prediction[rows]
        order = np.argsort(idx)
        have[airport] = np.full((val_counts[airport], *prediction.shape[1:]), np.nan, dtype=prediction.dtype) \
            if task == "forecast" else np.full((val_counts[airport],), -1, dtype=prediction.dtype)
        have[airport][idx[order]] = preds[order]

    missing: list[str] = []
    for airport in AIRPORT_ORDER:
        if task == "forecast":
            bad = np.isnan(have[airport]).all(axis=tuple(range(1, have[airport].ndim)))
        else:
            bad = have[airport] < 0
        missing.extend(_sample_id(airport, int(i)) for i in np.where(bad)[0])

    if missing and not allow_fill:
        raise AdapterError(
            f"{len(missing)} 个 val 样本无预测（评测器要求全覆盖，部分缺失行会被计分）。"
            f"正式 trial 必须由 D 侧全覆盖导出修复；仅接口联调可 allow_fill=True。"
            f"缺失示例: {missing[:5]}"
        )

    if missing:
        for airport in AIRPORT_ORDER:
            if task == "forecast":
                bad = np.isnan(have[airport]).all(axis=tuple(range(1, have[airport].ndim)))
                have[airport][bad] = FILL_VALUE_FORECAST
            else:
                have[airport][have[airport] < 0] = 0

    sample_ids = np.asarray(full_manifest_ids(val_counts), dtype="U128")
    pred = np.concatenate([have[a] for a in AIRPORT_ORDER], axis=0)

    if task == "forecast":
        if tuple(pred.shape[1:]) != FORECAST_TAIL_SHAPE:
            raise AdapterError(f"forecast pred 尾形状 {pred.shape[1:]} != {FORECAST_TAIL_SHAPE}")
        pred = pred.astype(np.float32)
        if np.isnan(pred).any() or np.isinf(pred).any():
            raise AdapterError("填充后仍存在 NaN/Inf")
        oob = int(((pred < 0.0) | (pred > 1.0)).sum())
        if oob:
            # 真实跑道位置越界：不修改、不拦截，交评测器按合同判 invalid；
            # 此处仅提醒（提交后 evaluator 将拒绝，trial 记 invalid 不入统计）
            import sys
            print(
                f"[adapter warning] {oob} 个真实跑道预测值越界 [0,1]，"
                f"原值保留提交（evaluator v1.0.1 容忍带内照常计分，超限判 invalid；禁止 clip）",
                file=sys.stderr,
            )
    else:
        pred = pred.astype(np.int64)
        if ((pred < 0) | (pred > 2)).any():
            raise AdapterError("classification 预测越界 {0,1,2}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, sample_id=sample_ids, pred=pred)
    return AdaptResult(
        out_path=out_path,
        n_rows=int(len(sample_ids)),
        n_missing_filled=len(missing),
        missing_sample_ids=tuple(missing),
    )
