"""A-6：harness 侧的冻结 evaluator 子进程客户端。

合同基准：C1 evaluator v1.0 冻结版实测行为（G-10 联调 8/27 校准，替代 12 文档草案期望）：
- CLI：--predictions / --trial-meta / --split val / --out-dir；
- 退出码：0=completed，2=invalid，3=failed；三种情况 metrics.json 均落盘，
  以 metrics.json 为唯一真相源（退出码只作交叉校验）；
- completed 的 metrics.json 键：status/task/endpoints(列表)/overall/anomaly_counts/
  contract_sha256/trial；endpoints 元素含 name/value/ci95/degenerate；
- invalid/failed = not_evaluated，绝不折算为候选被拒；
- evaluator 二进制与私有配置归 C 所有，本客户端只调用与解析，不 inspect 不修改。
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvaluationOutcome:
    status: str  # completed | invalid | failed
    metrics_by_endpoint: dict = field(default_factory=dict)  # name -> endpoint dict
    overall: dict = field(default_factory=dict)  # name -> overall dict
    anomaly_counts: dict = field(default_factory=dict)
    status_reason: str = ""
    evaluation_manifest: dict = field(default_factory=dict)
    raw_metrics: dict = field(default_factory=dict)


def run_evaluator(
    evaluator_cmd: list[str],
    predictions_path: Path,
    out_dir: Path,
    trial_meta_path: Path | None = None,
    split: str = "val",
    timeout_sec: int = 1800,
    cwd: Path | None = None,
    env: dict | None = None,
) -> EvaluationOutcome:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [*evaluator_cmd, "--predictions", str(predictions_path), "--split", split, "--out-dir", str(out_dir)]
    if trial_meta_path is not None:
        cmd += ["--trial-meta", str(trial_meta_path)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, check=False,
            cwd=str(cwd) if cwd else None, env=env,
        )
    except subprocess.TimeoutExpired:
        return EvaluationOutcome(status="failed", status_reason=f"evaluator timeout after {timeout_sec}s")
    except OSError as exc:
        return EvaluationOutcome(status="failed", status_reason=f"evaluator spawn error: {exc}")

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return EvaluationOutcome(
            status="failed",
            status_reason=f"no metrics.json (exit {proc.returncode}): {tail}",
        )
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return EvaluationOutcome(status="failed", status_reason=f"metrics.json malformed: {exc}")

    status = payload.get("status")
    if status == "invalid":
        return EvaluationOutcome(
            status="invalid",
            status_reason=str(payload.get("reason", "submission rejected by evaluator")),
            anomaly_counts=payload.get("anomaly_counts", {}),
            raw_metrics=payload,
        )
    if status == "failed":
        return EvaluationOutcome(
            status="failed",
            status_reason=str(payload.get("reason", "evaluator internal failure")),
            anomaly_counts=payload.get("anomaly_counts", {}),
            raw_metrics=payload,
        )
    if status != "completed":
        return EvaluationOutcome(status="failed", status_reason=f"unexpected status: {status!r}")
    if proc.returncode != 0:
        return EvaluationOutcome(
            status="failed",
            status_reason=f"metrics.json says completed but exit code {proc.returncode} (交叉校验失败)",
        )

    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        return EvaluationOutcome(status="failed", status_reason="completed status but endpoints missing or empty")
    metrics_by_endpoint = {}
    for ep in endpoints:
        if not isinstance(ep, dict) or "name" not in ep:
            return EvaluationOutcome(status="failed", status_reason="endpoint entry missing name")
        metrics_by_endpoint[ep["name"]] = ep
    overall = {o["name"]: o for o in payload.get("overall", []) if isinstance(o, dict) and "name" in o}

    manifest = {}
    manifest_path = out_dir / "evaluation_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return EvaluationOutcome(status="failed", status_reason="evaluation_manifest.json malformed")

    return EvaluationOutcome(
        status="completed",
        metrics_by_endpoint=metrics_by_endpoint,
        overall=overall,
        anomaly_counts=payload.get("anomaly_counts", {}),
        evaluation_manifest=manifest,
        raw_metrics=payload,
    )
