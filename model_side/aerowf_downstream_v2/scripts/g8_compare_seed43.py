#!/usr/bin/env python3
"""Compare seed43 original full_pipeline vs G-8 thinned rerun."""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/aerowf_downstream_v2/results/full_pipeline")
OLD = BASE / "seed43_v2"
NEW = BASE / "seed43_v2_g8_consistency"
OUT = Path("/root/autodl-tmp/aerowf_downstream_v2/results/analysis/g8_seed43_consistency.json")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cls_f1(metrics_path: Path) -> float:
    return float(load(metrics_path)["metrics"]["macro_f1"])


def fc_rmse(metrics_path: Path) -> float:
    m = load(metrics_path)["metrics"]
    vals = [m[h]["paper_table_aggregate"]["rmse_norm"] for h in ("T+1", "T+4", "T+8")]
    return sum(vals) / len(vals)


def pre_val(metrics_path: Path) -> float:
    return float(load(metrics_path)["trainer_metrics"]["val_loss"])


def main() -> int:
    if not (NEW / "pipeline_summary.json").is_file():
        print("G-8 seed43 summary missing")
        return 2
    pairs = {
        "pretrain_val_loss": (
            pre_val(OLD / "pretrain/metrics.json"),
            pre_val(NEW / "pretrain/metrics.json"),
        ),
        "forecast_scratch_rmse": (
            fc_rmse(OLD / "forecast_scratch/metrics.json"),
            fc_rmse(NEW / "forecast_scratch/metrics.json"),
        ),
        "forecast_pretrained_rmse": (
            fc_rmse(OLD / "forecast_pretrained/metrics.json"),
            fc_rmse(NEW / "forecast_pretrained/metrics.json"),
        ),
        "cls_scratch_macro_f1": (
            cls_f1(OLD / "classification_scratch/metrics.json"),
            cls_f1(NEW / "classification_scratch/metrics.json"),
        ),
        "cls_pretrained_macro_f1": (
            cls_f1(OLD / "classification_pretrained/metrics.json"),
            cls_f1(NEW / "classification_pretrained/metrics.json"),
        ),
    }
    rows = []
    max_abs = 0.0
    for name, (a, b) in pairs.items():
        diff = b - a
        max_abs = max(max_abs, abs(diff))
        rows.append({"metric": name, "original": a, "g8": b, "delta": diff})
    verdict = "bit_or_float_noise" if max_abs < 1e-9 else (
        "float_noise_ok" if max_abs < 1e-6 else "BEHAVIOR_CHANGE"
    )
    report = {
        "verdict": verdict,
        "max_abs_delta": max_abs,
        "threshold_fail": 1e-9,
        "pairs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if verdict != "BEHAVIOR_CHANGE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
