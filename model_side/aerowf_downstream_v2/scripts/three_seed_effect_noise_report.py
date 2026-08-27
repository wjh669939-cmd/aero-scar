#!/usr/bin/env python3
"""Three-seed paired effect-size vs noise report (Contract V2, validation only)."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE = Path("/root/autodl-tmp/aerowf_downstream_v2")
REPORT_DIR = WORKSPACE / "results" / "analysis"
SEEDS = [42, 43, 2027]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_paths(seed: int) -> dict[str, Path]:
    if seed == 42:
        base = Path("/root/autodl-tmp")
        return {
            "pretrain": base
            / "aerowf_baseline/AeroWF/results/aerowf_unified_pretrain_full_formal_seed42_v1/metrics.json",
            "cls_scratch": base
            / "aerowf_downstream_v2/results/classification/scratch_formal_seed42_v2/metrics.json",
            "cls_pretrained": base
            / "aerowf_downstream_v2/results/classification/pretrained_formal_seed42_v2/metrics.json",
            "fc_scratch": base
            / "aerowf_downstream_v2/results/forecast/scratch_formal_seed42_v2/metrics.json",
            "fc_pretrained": base
            / "aerowf_downstream_v2/results/forecast/pretrained_formal_seed42_v2/metrics.json",
            "pipeline_summary": None,
        }
    root = WORKSPACE / "results" / "full_pipeline" / f"seed{seed}_v2"
    return {
        "pretrain": root / "pretrain/metrics.json",
        "cls_scratch": root / "classification_scratch/metrics.json",
        "cls_pretrained": root / "classification_pretrained/metrics.json",
        "fc_scratch": root / "forecast_scratch/metrics.json",
        "fc_pretrained": root / "forecast_pretrained/metrics.json",
        "pipeline_summary": root / "pipeline_summary.json",
    }


def cls_metrics(path: Path) -> dict[str, Any]:
    m = load_json(path)
    met = m["metrics"]
    pc = met["per_class"]
    return {
        "status": m.get("status"),
        "epochs": m.get("epochs_completed"),
        "best_epoch": m.get("best_epoch"),
        "macro_f1": met["macro_f1"],
        "CSI_macro": met["CSI_macro"],
        "accuracy": met["accuracy"],
        "GOOD_f1": pc["GOOD"]["f1"],
        "PRECIP_f1": pc["PRECIP"]["f1"],
        "HAZARD_f1": pc["HAZARD"]["f1"],
    }


def fc_macro(path: Path) -> dict[str, Any]:
    m = load_json(path)
    metrics = m.get("metrics", {})
    maes, rmses = [], []
    horizon: dict[str, dict[str, float]] = {}
    for h in ("T+1", "T+4", "T+8"):
        if h not in metrics:
            continue
        pa = metrics[h]["paper_table_aggregate"]
        horizon[h] = {
            "mae_norm": pa["mae_norm"],
            "rmse_norm": pa["rmse_norm"],
        }
        maes.append(pa["mae_norm"])
        rmses.append(pa["rmse_norm"])
    return {
        "status": m.get("status"),
        "epochs": m.get("epochs_completed"),
        "best_epoch": m.get("best_epoch"),
        "best_val_mse_norm": m.get("best_val_mse_norm"),
        "MAE_macro_norm": sum(maes) / len(maes),
        "RMSE_macro_norm": sum(rmses) / len(rmses),
        "horizon": horizon,
    }


def pretrain_metrics(path: Path) -> dict[str, Any]:
    m = load_json(path)
    tm = m.get("trainer_metrics", {})
    return {
        "status": m.get("status"),
        "epochs": m.get("epochs"),
        "val_loss": tm.get("val_loss"),
        "train_loss": tm.get("train_loss"),
    }


def mean(xs: list[float]) -> float:
    return statistics.mean(xs)


def stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def paired_summary(
    name: str,
    deltas: list[float],
    *,
    higher_better: bool,
    unit: str,
) -> dict[str, Any]:
    mu = mean(deltas)
    sd = stdev(deltas)
    snr = abs(mu) / sd if sd > 1e-12 else float("inf")
    rng = max(deltas) - min(deltas)
    sign_consistent = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
    if snr >= 1.5 and sign_consistent:
        verdict = "效应主导"
    elif snr >= 0.8 and sign_consistent:
        verdict = "效应可见但噪声不小"
    elif sign_consistent:
        verdict = "方向一致但效应弱于噪声"
    else:
        verdict = "噪声主导（符号不一致）"

    return {
        "metric": name,
        "unit": unit,
        "higher_better_for_positive_delta": higher_better,
        "paired_deltas_by_seed": deltas,
        "mean_delta": mu,
        "sd_delta": sd,
        "range_delta": rng,
        "snr_abs_mean_over_sd": snr,
        "sign_consistent_across_seeds": sign_consistent,
        "verdict": verdict,
    }


def collect_seed(seed: int) -> dict[str, Any]:
    paths = seed_paths(seed)
    missing = [k for k, p in paths.items() if p is not None and not p.is_file()]
    if missing:
        return {"seed": seed, "ready": False, "missing": missing}

    cls_s = cls_metrics(paths["cls_scratch"])
    cls_p = cls_metrics(paths["cls_pretrained"])
    fc_s = fc_macro(paths["fc_scratch"])
    fc_p = fc_macro(paths["fc_pretrained"])
    pt = pretrain_metrics(paths["pretrain"])

    return {
        "seed": seed,
        "ready": True,
        "pretrain": pt,
        "classification": {
            "scratch": cls_s,
            "pretrained": cls_p,
            "paired_delta": {
                "macro_f1": cls_p["macro_f1"] - cls_s["macro_f1"],
                "CSI_macro": cls_p["CSI_macro"] - cls_s["CSI_macro"],
                "accuracy": cls_p["accuracy"] - cls_s["accuracy"],
                "GOOD_f1": cls_p["GOOD_f1"] - cls_s["GOOD_f1"],
                "PRECIP_f1": cls_p["PRECIP_f1"] - cls_s["PRECIP_f1"],
                "HAZARD_f1": cls_p["HAZARD_f1"] - cls_s["HAZARD_f1"],
            },
        },
        "forecast": {
            "scratch": fc_s,
            "pretrained": fc_p,
            "paired_delta": {
                # Pre − Scr on error metrics: positive => scratch wins (pretrain hurts)
                "RMSE_macro_norm": fc_p["RMSE_macro_norm"] - fc_s["RMSE_macro_norm"],
                "MAE_macro_norm": fc_p["MAE_macro_norm"] - fc_s["MAE_macro_norm"],
                "best_val_mse_norm": fc_p["best_val_mse_norm"] - fc_s["best_val_mse_norm"],
            },
        },
    }


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [r for r in rows if r.get("ready")]
    seeds_ready = [r["seed"] for r in ready]

    def deltas(task: str, key: str) -> list[float]:
        return [r[task]["paired_delta"][key] for r in ready]

    cls_pairs = {
        "macro_f1": paired_summary(
            "Classification Macro-F1 (Pretrained − Scratch)",
            deltas("classification", "macro_f1"),
            higher_better=True,
            unit="F1 points",
        ),
        "CSI_macro": paired_summary(
            "Classification CSI_macro (Pretrained − Scratch)",
            deltas("classification", "CSI_macro"),
            higher_better=True,
            unit="CSI points",
        ),
        "HAZARD_f1": paired_summary(
            "Classification HAZARD F1 (Pretrained − Scratch)",
            deltas("classification", "HAZARD_f1"),
            higher_better=True,
            unit="F1 points",
        ),
    }
    fc_pairs = {
        "RMSE_macro_norm": paired_summary(
            "Forecast RMSE_macro_norm (Pretrained − Scratch)",
            deltas("forecast", "RMSE_macro_norm"),
            higher_better=True,
            unit="normalized RMSE",
        ),
        "MAE_macro_norm": paired_summary(
            "Forecast MAE_macro_norm (Pretrained − Scratch)",
            deltas("forecast", "MAE_macro_norm"),
            higher_better=True,
            unit="normalized MAE",
        ),
    }

    scratch_cls = [r["classification"]["scratch"]["macro_f1"] for r in ready]
    pre_cls = [r["classification"]["pretrained"]["macro_f1"] for r in ready]
    scratch_fc = [r["forecast"]["scratch"]["RMSE_macro_norm"] for r in ready]
    pre_fc = [r["forecast"]["pretrained"]["RMSE_macro_norm"] for r in ready]

    noise_floor = {
        "classification_scratch_macro_f1_sd": stdev(scratch_cls),
        "classification_pretrained_macro_f1_sd": stdev(pre_cls),
        "forecast_scratch_rmse_sd": stdev(scratch_fc),
        "forecast_pretrained_rmse_sd": stdev(pre_fc),
        "pretrain_val_loss_sd": stdev([r["pretrain"]["val_loss"] for r in ready]),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds_requested": SEEDS,
        "seeds_ready": seeds_ready,
        "n_seeds": len(seeds_ready),
        "methodology": {
            "design": "paired_within_seed",
            "classification_delta": "Pretrained − Scratch (positive = pretrain helps)",
            "forecast_delta": "Pretrained − Scratch on RMSE/MAE (positive = scratch better)",
            "snr_definition": "|mean(Δ across seeds)| / stdev(Δ across seeds)",
            "scope": "validation_only_not_sealed_test",
            "protocol": "Contract V2",
        },
        "per_seed": ready,
        "paired_effect_vs_noise": {
            "classification": cls_pairs,
            "forecast": fc_pairs,
        },
        "cross_seed_noise_floor": noise_floor,
        "candidate_selection": {
            "classification": "pretrained if mean(macro_f1 delta) > 0",
            "forecast": "scratch if mean(RMSE delta) > 0",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 三 Seed 配对报告：效应量 vs 噪声",
        "",
        f"- 生成时间 (UTC): {report['generated_at_utc']}",
        f"- 可用 seed: {report['seeds_ready']} / {report['seeds_requested']}",
        f"- 口径: {report['methodology']['design']} · {report['methodology']['protocol']}",
        "",
        "## 1. 配对定义",
        "",
        "| 任务 | 配对差分 Δ | 正号含义 |",
        "|------|-----------|---------|",
        "| 分类 | Pretrained − Scratch | 预训练有正迁移 |",
        "| 预测 | Pretrained − Scratch (RMSE/MAE) | Scratch 更优（Pre 误差更大） |",
        "",
        "SNR = |mean(Δ)| / SD(Δ)，跨 seed 共 n={}。".format(report["n_seeds"]),
        "",
        "## 2. 逐 Seed 原始值",
        "",
        "| Seed | Cls Scratch F1 | Cls Pre F1 | ΔF1 | Fc Scr RMSE | Fc Pre RMSE | ΔRMSE | Pretrain val_loss |",
        "|------|----------------|------------|-----|-------------|-------------|-------|-------------------|",
    ]
    for r in report["per_seed"]:
        cs = r["classification"]["scratch"]["macro_f1"]
        cp = r["classification"]["pretrained"]["macro_f1"]
        df = r["classification"]["paired_delta"]["macro_f1"]
        fs = r["forecast"]["scratch"]["RMSE_macro_norm"]
        fp = r["forecast"]["pretrained"]["RMSE_macro_norm"]
        dr = r["forecast"]["paired_delta"]["RMSE_macro_norm"]
        vl = r["pretrain"]["val_loss"]
        lines.append(
            f"| {r['seed']} | {cs:.4f} | {cp:.4f} | {df:+.4f} | {fs:.5f} | {fp:.5f} | {dr:+.5f} | {vl:.6f} |"
        )

    lines += ["", "## 3. 效应量 vs 噪声（跨 seed 汇总）", ""]
    for section in ("classification", "forecast"):
        lines.append(f"### {section.title()}")
        lines.append("")
        lines.append("| 指标 | mean(Δ) | SD(Δ) | range(Δ) | SNR | 符号一致 | 判定 |")
        lines.append("|------|---------|-------|----------|-----|---------|------|")
        for item in report["paired_effect_vs_noise"][section].values():
            deltas = item["paired_deltas_by_seed"]
            delta_str = ", ".join(f"{d:+.4f}" for d in deltas)
            lines.append(
                f"| {item['metric']} | {item['mean_delta']:+.4f} | {item['sd_delta']:.4f} | "
                f"{item['range_delta']:.4f} | {item['snr_abs_mean_over_sd']:.2f} | "
                f"{'是' if item['sign_consistent_across_seeds'] else '否'} | {item['verdict']} |"
            )
            lines.append(f"| ↳ 各 seed Δ | {delta_str} | | | | | |")
        lines.append("")

    nf = report["cross_seed_noise_floor"]
    lines += [
        "## 4. 跨 seed 噪声底（未配对 baseline 波动）",
        "",
        "| 指标 | SD across seeds |",
        "|------|-----------------|",
        f"| 分类 Scratch Macro-F1 | {nf['classification_scratch_macro_f1_sd']:.4f} |",
        f"| 分类 Pretrained Macro-F1 | {nf['classification_pretrained_macro_f1_sd']:.4f} |",
        f"| 预测 Scratch RMSE | {nf['forecast_scratch_rmse_sd']:.5f} |",
        f"| 预测 Pretrained RMSE | {nf['forecast_pretrained_rmse_sd']:.5f} |",
        f"| 预训练 val_loss | {nf['pretrain_val_loss_sd']:.6f} |",
        "",
        "## 5. 解读要点",
        "",
        "- **分类**：若 mean(ΔF1) 显著大于 SD(ΔF1) 且三 seed 同号，则预训练正迁移是稳定信号。",
        "- **预测**：若 Scratch−Pretrained 的 mean(ΔRMSE) 接近 0 或 SNR<1，则预训练对预测无稳定收益。",
        "- n=3 时 SD 估计粗糙，本报告用于方向性判断，不作严格显著性检验。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [collect_seed(s) for s in SEEDS]
    not_ready = [r for r in rows if not r.get("ready")]
    if not_ready:
        status_path = REPORT_DIR / "three_seed_report_status.json"
        status_path.write_text(
            json.dumps(
                {
                    "ready": False,
                    "missing": not_ready,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Not all seeds ready. Missing: {not_ready}")
        return 2

    report = build_report(rows)
    json_path = REPORT_DIR / "three_seed_effect_noise_report.json"
    md_path = REPORT_DIR / "three_seed_effect_noise_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
