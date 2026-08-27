"""G-7：CLH trial → aerowf_full_pipeline_v2.py 子进程执行器。

判定合同 = D 侧 HANDOFF.md §6（八条失败线 + 六条成功线），要点：
- 不解析终端文字判成败，只看退出码 + 落盘产物；
- 六条成功条件同时满足才算 success：
  1. 进程退出码 0
  2. pipeline_summary.json 存在
  3. 五阶段 metrics.json 的 status 均为 success
  4. 五阶段 test_used 均为 false
  5. 所有 best_model.pth 存在
  6. pretrained 阶段 checkpoint_load 的 missing_keys/unexpected_keys 均为空
- 附加失败线：指标出现 NaN/Inf；输出目录与已有实验冲突（非空目录直接拒绝，
  不覆盖不复用，保留现场换新 RUN_ID）；
- 失败 trial 保留 stdout/stderr/config/输出目录现场，绝不进入候选比较。
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

STAGES = (
    "pretrain",
    "forecast_scratch",
    "forecast_pretrained",
    "classification_scratch",
    "classification_pretrained",
)
PRETRAINED_STAGES = ("forecast_pretrained", "classification_pretrained")


@dataclass(frozen=True)
class PipelineConfig:
    """执行环境（部署到 D 机器时按 HANDOFF §2/§4 填实）。"""

    pipeline_script: Path
    workdir: Path  # 必须从 aerowf_baseline/AeroWF 启动（HANDOFF §4）
    output_root_base: Path  # 例如 results/harness/
    python_bin: str = "python"
    batch_size: int = 128
    pretrain_epochs: int = 100
    downstream_epochs: int = 30
    patience: int = 10
    min_delta: float = 1e-4
    num_workers: int = 0
    timeout_sec: int = 6 * 3600  # 正式全流程 ~2.7h，留一倍余量


@dataclass(frozen=True)
class ExecutionOutcome:
    status: str  # success | failed
    run_id: str
    output_root: Path
    failure_reasons: list[str] = field(default_factory=list)
    stage_metrics: dict = field(default_factory=dict)  # stage -> metrics.json 内容
    pipeline_summary: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    checkpoint_sha256: dict = field(default_factory=dict)  # stage -> best_model.pth 摘要


def make_run_id(task: str, seed: int) -> str:
    """HANDOFF §10 建议格式：<task>_<timestamp>_<seed>_<short_hash>。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{task}_{ts}_seed{seed}_{short}"


def _has_nonfinite(node: object) -> bool:
    if isinstance(node, float):
        return not math.isfinite(node)
    if isinstance(node, dict):
        return any(_has_nonfinite(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_nonfinite(v) for v in node)
    return False


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while blob := fh.read(chunk):
            h.update(blob)
    return h.hexdigest()


def judge_output_tree(output_root: Path, exit_code: int) -> tuple[list[str], dict, dict]:
    """六条件判定，与执行解耦以便单测和事后复核。

    返回 (failure_reasons, stage_metrics, pipeline_summary)。
    """
    reasons: list[str] = []
    stage_metrics: dict = {}
    summary: dict = {}

    if exit_code != 0:
        reasons.append(f"非零退出码: {exit_code}")

    summary_path = output_root / "pipeline_summary.json"
    if not summary_path.exists():
        reasons.append("缺少 pipeline_summary.json")
    else:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            reasons.append(f"pipeline_summary.json 解析失败: {exc}")

    for stage in STAGES:
        mpath = output_root / stage / "metrics.json"
        if not mpath.exists():
            reasons.append(f"{stage}: 缺少 metrics.json")
            continue
        try:
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            reasons.append(f"{stage}: metrics.json 解析失败: {exc}")
            continue
        stage_metrics[stage] = metrics

        if metrics.get("status") != "success":
            reasons.append(f"{stage}: status={metrics.get('status')!r} 非 success")
        if metrics.get("test_used") is not False:
            reasons.append(f"{stage}: test_used={metrics.get('test_used')!r} 必须为 false")
        if _has_nonfinite(metrics):
            reasons.append(f"{stage}: 指标含 NaN/Inf")
        if not (output_root / stage / "checkpoints" / "best_model.pth").exists():
            reasons.append(f"{stage}: 缺少 checkpoints/best_model.pth")

    for stage in PRETRAINED_STAGES:
        metrics = stage_metrics.get(stage)
        if metrics is None:
            continue  # 缺 metrics 已计入
        load = metrics.get("checkpoint_load")
        if not isinstance(load, dict):
            reasons.append(f"{stage}: 缺少 checkpoint_load 记录")
            continue
        if load.get("missing_keys") or load.get("unexpected_keys"):
            reasons.append(
                f"{stage}: checkpoint_load keys 非空 "
                f"(missing={len(load.get('missing_keys') or [])}, "
                f"unexpected={len(load.get('unexpected_keys') or [])})"
            )

    return reasons, stage_metrics, summary


def run_pipeline(
    config: PipelineConfig,
    seed: int,
    run_id: str | None = None,
    extra_args: list[str] | None = None,
) -> ExecutionOutcome:
    """启动一次全流程并按六条件判定。

    - 输出目录已存在且非空 → 直接 failed（HANDOFF §10：保留现场，换新 RUN_ID 重跑）；
    - stdout/stderr 全程落盘到 output_root 同级的 <run_id>.stdout/.stderr，
      失败时现场完整保留。
    """
    rid = run_id or make_run_id("trial", seed)
    output_root = config.output_root_base / rid

    if output_root.exists() and any(output_root.iterdir()):
        return ExecutionOutcome(
            status="failed",
            run_id=rid,
            output_root=output_root,
            failure_reasons=[f"输出目录非空，与已有实验冲突: {output_root}"],
        )
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        config.python_bin,
        "-u",
        str(config.pipeline_script),
        "--seed", str(seed),
        "--batch-size", str(config.batch_size),
        "--pretrain-epochs", str(config.pretrain_epochs),
        "--downstream-epochs", str(config.downstream_epochs),
        "--patience", str(config.patience),
        "--min-delta", str(config.min_delta),
        "--num-workers", str(config.num_workers),
        "--output-root", str(output_root),
        *(extra_args or []),
    ]

    stdout_path = config.output_root_base / f"{rid}.stdout"
    stderr_path = config.output_root_base / f"{rid}.stderr"
    started = time.monotonic()
    try:
        with stdout_path.open("w") as out_fh, stderr_path.open("w") as err_fh:
            proc = subprocess.run(
                cmd,
                cwd=str(config.workdir),
                stdout=out_fh,
                stderr=err_fh,
                timeout=config.timeout_sec,
                check=False,
            )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        return ExecutionOutcome(
            status="failed",
            run_id=rid,
            output_root=output_root,
            failure_reasons=[f"超时（>{config.timeout_sec}s），进程已终止，现场保留"],
            elapsed_sec=time.monotonic() - started,
        )
    except OSError as exc:
        return ExecutionOutcome(
            status="failed",
            run_id=rid,
            output_root=output_root,
            failure_reasons=[f"进程启动失败: {exc}"],
            elapsed_sec=time.monotonic() - started,
        )
    elapsed = time.monotonic() - started

    reasons, stage_metrics, summary = judge_output_tree(output_root, exit_code)

    checkpoint_sha: dict = {}
    if not reasons:
        for stage in STAGES:
            ckpt = output_root / stage / "checkpoints" / "best_model.pth"
            checkpoint_sha[stage] = _sha256(ckpt)

    return ExecutionOutcome(
        status="success" if not reasons else "failed",
        run_id=rid,
        output_root=output_root,
        failure_reasons=reasons,
        stage_metrics=stage_metrics,
        pipeline_summary=summary,
        elapsed_sec=elapsed,
        checkpoint_sha256=checkpoint_sha,
    )
