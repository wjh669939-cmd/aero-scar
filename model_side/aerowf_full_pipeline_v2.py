#!/usr/bin/env python3
"""Run the frozen-data AeroWF V2 model-side baseline end to end.

Stages:
1. unified pretraining on the frozen PRE2020 train/validation release;
2. V2 Forecast from scratch;
3. V2 Forecast initialized from the new pretraining checkpoint;
4. V2 Classification from scratch;
5. V2 Classification initialized from the new pretraining checkpoint;
6. validation-only aggregation and candidate freeze.

The controller is resume-safe: a stage is skipped only when its metrics.json
exists and reports status=success.  A non-empty incomplete stage directory is
treated as an error so partial outputs are never overwritten silently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--downstream-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/root/autodl-tmp/aerowf_baseline/AeroWF"),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("/root/autodl-tmp/aerowf_downstream_v2"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--o2-event-flags",
        type=Path,
        default=None,
        help="转发给 forecast 两阶段的 O2 事件标志 CSV（train-only；缺省行为不变）",
    )
    parser.add_argument(
        "--reuse-pretrain-checkpoint",
        type=Path,
        default=None,
        help=(
            "跳过预训练阶段，复用给定 best_model.pth（仅限编辑不触及预训练代码的 trial，"
            "即 R/O-tier1 轴；tier2 禁用）。权重级等价依据：独立重训的 seed42 checkpoint "
            "与正式 parent 212/212 张量逐位一致（2026-08-28 验证）。"
        ),
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def successful_metrics(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "metrics.json"
    if not path.is_file():
        return None
    result = read_json(path)
    return result if result.get("status") == "success" else None


def ensure_clean_or_success(output_dir: Path) -> dict[str, Any] | None:
    result = successful_metrics(output_dir)
    if result is not None:
        return result
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete non-empty stage directory must be reviewed manually: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return None


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_stage(
    name: str,
    command: list[str],
    output_dir: Path,
    cwd: Path,
) -> dict[str, Any]:
    existing = ensure_clean_or_success(output_dir)
    if existing is not None:
        print(f"[{name}] already successful; skipping", flush=True)
        return existing

    record = {
        "stage": name,
        "status": "running",
        "started_at_utc": now_iso(),
        "cwd": str(cwd),
        "command": command,
        "command_text": command_text(command),
    }
    (output_dir / "stage_command.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{'=' * 78}\n[{name}] START\n{record['command_text']}\n", flush=True)

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    start = time.time()
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    record.update(
        {
            "finished_at_utc": now_iso(),
            "elapsed_seconds": time.time() - start,
            "return_code": return_code,
            "status": "success" if return_code == 0 else "failed",
        }
    )
    (output_dir / "stage_command.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if return_code != 0:
        raise RuntimeError(f"Stage {name} failed with return code {return_code}")

    result = successful_metrics(output_dir)
    if result is None:
        raise RuntimeError(f"Stage {name} returned zero but has no successful metrics.json")
    print(f"[{name}] SUCCESS", flush=True)
    return result


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    workspace_root = args.workspace_root.resolve()
    source_root = workspace_root / "src"
    contract = workspace_root / "contracts" / "DOWNSTREAM_TASK_CONTRACT_v2.json"
    pretrain_script = repo_root / "aerowf_unified_pretrain_train_v2.py"
    forecast_script = source_root / "aerowf_forecast_train_v2.py"
    classification_script = source_root / "aerowf_classification_train_v2.py"
    persistence_metrics = (
        workspace_root / "results" / "forecast" / "persistence_v2" / "metrics.json"
    )

    if args.output_root is None:
        output_root = (
            workspace_root
            / "results"
            / "full_pipeline"
            / f"seed{args.seed}_v2"
        )
    else:
        output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for required in (
        pretrain_script,
        forecast_script,
        classification_script,
        contract,
    ):
        require_file(required)

    python = sys.executable
    pretrain_dir = output_root / "pretrain"
    forecast_scratch_dir = output_root / "forecast_scratch"
    forecast_pretrained_dir = output_root / "forecast_pretrained"
    classification_scratch_dir = output_root / "classification_scratch"
    classification_pretrained_dir = output_root / "classification_pretrained"
    new_pretrain_checkpoint = pretrain_dir / "checkpoints" / "best_model.pth"

    pipeline_config = {
        "schema_version": "2.0",
        "pipeline": "AeroWF_frozen_data_model_side_full_baseline",
        "scope": "pretraining_plus_forecast_plus_classification_validation",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "pretrain_epochs": args.pretrain_epochs,
        "downstream_epochs": args.downstream_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "num_workers": args.num_workers,
        "test_used": False,
        "raw_preprocessing_included": False,
        "frozen_release_used": True,
        "created_at_utc": now_iso(),
        "paths": {
            "repo_root": str(repo_root),
            "workspace_root": str(workspace_root),
            "output_root": str(output_root),
        },
        "source_sha256": {
            "pretrain": sha256_file(pretrain_script),
            "forecast": sha256_file(forecast_script),
            "classification": sha256_file(classification_script),
            "contract": sha256_file(contract),
        },
    }
    (output_root / "pipeline_config.json").write_text(
        json.dumps(pipeline_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pipeline_start = time.time()
    if args.reuse_pretrain_checkpoint is not None:
        source_checkpoint = args.reuse_pretrain_checkpoint
        require_file(source_checkpoint)
        source_metrics = source_checkpoint.parent.parent / "metrics.json"
        require_file(source_metrics)
        (pretrain_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, new_pretrain_checkpoint)
        pretrain = json.loads(source_metrics.read_text(encoding="utf-8"))
        # 溯源标记：这些 metrics 是被复用权重的真实训练记录，非本次训练产出
        pretrain["reused_from_checkpoint"] = str(source_checkpoint)
        pretrain["reused_source_sha256"] = sha256_file(source_checkpoint)
        (pretrain_dir / "metrics.json").write_text(
            json.dumps(pretrain, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pretrain_dir / "stage_command.json").write_text(
            json.dumps(
                {
                    "stage": "pretrain",
                    "status": "reused_checkpoint",
                    "source_checkpoint": str(source_checkpoint),
                    "source_sha256": pretrain["reused_source_sha256"],
                    "basis": "weight-level equivalence verified 2026-08-28 (212/212 tensors bit-identical across independent seed-42 reruns)",
                    "at_utc": now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("[pretrain] REUSED checkpoint:", source_checkpoint, flush=True)
    else:
        pretrain = run_stage(
            "pretrain",
            [
                python,
                "-u",
                str(pretrain_script),
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.pretrain_epochs),
                "--patience",
                str(args.patience),
                "--min-delta",
                str(args.min_delta),
                "--num-workers",
                str(args.num_workers),
                "--formal",
                "--output-dir",
                str(pretrain_dir),
            ],
            pretrain_dir,
            repo_root,
        )
    require_file(new_pretrain_checkpoint)

    forecast_extra = (
        ["--o2-event-flags", str(args.o2_event_flags)] if args.o2_event_flags else []
    )
    common_downstream = [
        "--epochs",
        str(args.downstream_epochs),
        "--formal",
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--weight-decay",
        "1e-4",
        "--grad-clip",
        "3.0",
        "--checkpoint",
        str(new_pretrain_checkpoint),
    ]

    forecast_scratch = run_stage(
        "forecast_scratch",
        [
            python,
            "-u",
            str(forecast_script),
            "--initialization",
            "scratch",
            "--scratch-lr",
            "1e-4",
            *common_downstream,
            *forecast_extra,
            "--min-delta",
            "1e-5",
            "--output-dir",
            str(forecast_scratch_dir),
        ],
        forecast_scratch_dir,
        repo_root,
    )
    forecast_pretrained = run_stage(
        "forecast_pretrained",
        [
            python,
            "-u",
            str(forecast_script),
            "--initialization",
            "pretrained",
            "--encoder-lr",
            "1e-5",
            "--head-lr",
            "1e-4",
            *common_downstream,
            *forecast_extra,
            "--min-delta",
            "1e-5",
            "--output-dir",
            str(forecast_pretrained_dir),
        ],
        forecast_pretrained_dir,
        repo_root,
    )
    classification_scratch = run_stage(
        "classification_scratch",
        [
            python,
            "-u",
            str(classification_script),
            "--initialization",
            "scratch",
            "--scratch-lr",
            "1e-4",
            *common_downstream,
            "--min-delta",
            str(args.min_delta),
            "--output-dir",
            str(classification_scratch_dir),
        ],
        classification_scratch_dir,
        repo_root,
    )
    classification_pretrained = run_stage(
        "classification_pretrained",
        [
            python,
            "-u",
            str(classification_script),
            "--initialization",
            "pretrained",
            "--encoder-lr",
            "1e-5",
            "--head-lr",
            "1e-4",
            *common_downstream,
            "--min-delta",
            str(args.min_delta),
            "--output-dir",
            str(classification_pretrained_dir),
        ],
        classification_pretrained_dir,
        repo_root,
    )

    fs = forecast_scratch["metrics"]["summary"]
    fp = forecast_pretrained["metrics"]["summary"]
    cs = classification_scratch["metrics"]
    cp = classification_pretrained["metrics"]
    selected_forecast = (
        "pretrained"
        if fp["RMSE_macro_norm"] < fs["RMSE_macro_norm"]
        else "scratch"
    )
    selected_classification = (
        "pretrained" if cp["macro_f1"] > cs["macro_f1"] else "scratch"
    )

    stage_dirs = {
        "pretrain": pretrain_dir,
        "forecast_scratch": forecast_scratch_dir,
        "forecast_pretrained": forecast_pretrained_dir,
        "classification_scratch": classification_scratch_dir,
        "classification_pretrained": classification_pretrained_dir,
    }
    artifacts: dict[str, Any] = {}
    for name, directory in stage_dirs.items():
        metric_path = directory / "metrics.json"
        checkpoint_path = directory / "checkpoints" / "best_model.pth"
        artifacts[name] = {
            "metrics": str(metric_path),
            "metrics_sha256": sha256_file(metric_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }

    summary = {
        "status": "success",
        **pipeline_config,
        "completed_at_utc": now_iso(),
        "elapsed_seconds": time.time() - pipeline_start,
        "pretraining": {
            "best_checkpoint": str(new_pretrain_checkpoint),
            "best_checkpoint_sha256": sha256_file(new_pretrain_checkpoint),
            "trainer_metrics": pretrain["trainer_metrics"],
        },
        "forecast": {
            "scratch": fs,
            "pretrained": fp,
            "pretrained_minus_scratch": {
                "MAE_macro_norm": fp["MAE_macro_norm"] - fs["MAE_macro_norm"],
                "RMSE_macro_norm": fp["RMSE_macro_norm"] - fs["RMSE_macro_norm"],
            },
            "selected_validation_candidate": selected_forecast,
        },
        "classification": {
            "majority": classification_scratch["majority_validation_baseline"],
            "scratch": {
                "accuracy": cs["accuracy"],
                "macro_f1": cs["macro_f1"],
                "CSI_macro": cs["CSI_macro"],
            },
            "pretrained": {
                "accuracy": cp["accuracy"],
                "macro_f1": cp["macro_f1"],
                "CSI_macro": cp["CSI_macro"],
            },
            "pretrained_minus_scratch": {
                "accuracy": cp["accuracy"] - cs["accuracy"],
                "macro_f1": cp["macro_f1"] - cs["macro_f1"],
                "CSI_macro": cp["CSI_macro"] - cs["CSI_macro"],
            },
            "selected_validation_candidate": selected_classification,
        },
        "persistence_reference": (
            {
                "path": str(persistence_metrics),
                "sha256": sha256_file(persistence_metrics),
            }
            if persistence_metrics.is_file()
            else None
        ),
        "artifacts": artifacts,
        "candidate_freeze": {
            "scope": "validation_only_not_sealed_test",
            "forecast": selected_forecast,
            "classification": selected_classification,
            "test_used": False,
        },
    }
    summary_path = output_root / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = [
        {
            "task": "forecast",
            "method": "scratch",
            "primary_metric": "RMSE_macro_norm",
            "value": fs["RMSE_macro_norm"],
            "secondary_metric": "MAE_macro_norm",
            "secondary_value": fs["MAE_macro_norm"],
        },
        {
            "task": "forecast",
            "method": "pretrained",
            "primary_metric": "RMSE_macro_norm",
            "value": fp["RMSE_macro_norm"],
            "secondary_metric": "MAE_macro_norm",
            "secondary_value": fp["MAE_macro_norm"],
        },
        {
            "task": "classification",
            "method": "scratch",
            "primary_metric": "macro_f1",
            "value": cs["macro_f1"],
            "secondary_metric": "CSI_macro",
            "secondary_value": cs["CSI_macro"],
        },
        {
            "task": "classification",
            "method": "pretrained",
            "primary_metric": "macro_f1",
            "value": cp["macro_f1"],
            "secondary_metric": "CSI_macro",
            "secondary_value": cp["CSI_macro"],
        },
    ]
    with (output_root / "pipeline_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 78}\nFULL PIPELINE SUCCESS")
    print(json.dumps(summary["candidate_freeze"], ensure_ascii=False, indent=2))
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
