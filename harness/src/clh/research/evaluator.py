"""Independent evaluator. Metrics, splits, and test labels are not agent-editable.

Custody model (P0 fix, 2026-08-26): agent-authored trial code NEVER runs inside
this process. Trials execute via ``clh.research.subproc`` in an isolated
subprocess with a whitelisted environment; this process only serializes
label-safe inputs and scores the returned predictions. Data-axis extra sources
are taken from external_manifest.json (parsed as JSON here) plus data.py's
extra_source_ids() probed in the same isolated subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clh.config import HarnessConfig
from clh.core.errors import EvaluatorError
from clh.domain.dummy.weather import SplitName, SyntheticAirportWeather
from clh.domain.protocol import DummyATCAdapter
from clh.research.cards import MetricsBundle
from clh.research.reward import aggregate_improvement, robust_score, safety_ok
from clh.research.subproc import probe_extra_source_ids, run_pipeline_subprocess


class IndependentEvaluator:
    """Harness-owned scorer. The agent submits a workspace; it never sees test y."""

    def __init__(
        self,
        config: HarnessConfig,
        world: Any,
        baseline_workspace: Path,
    ) -> None:
        self.config = config
        self.baseline_workspace = baseline_workspace
        self.leakage_log: list[dict[str, Any]] = []
        if isinstance(world, SyntheticAirportWeather):
            adapter = DummyATCAdapter.__new__(DummyATCAdapter)
            adapter.config = config
            adapter._weather = world
            self.adapter = adapter
        else:
            self.adapter = world
        self.baseline = self._score(baseline_workspace, "val")

    def evaluate_workspace(self, workspace: Path, *, split: SplitName = "val") -> MetricsBundle:
        if split != "val":
            raise EvaluatorError("search loop may only score the visible val split")
        return self._score(workspace, split)

    def certify_workspace(self, workspace: Path, *, split: str) -> MetricsBundle:
        if not str(split).startswith("test"):
            raise EvaluatorError("certification must use a hidden test split")
        return self._score(workspace, split)

    def compare(self, candidate: MetricsBundle, baseline: MetricsBundle | None = None) -> dict[str, Any]:
        base = baseline or self.baseline
        improvement = aggregate_improvement(candidate, base)
        return {
            "improvement": improvement,
            "robust_score": robust_score(candidate, base),
            "safety_ok": safety_ok(candidate, base, self.config.research.safety_csi_tolerance),
            "delta_mae": base.mae - candidate.mae,
            "delta_csi": candidate.hazard_csi - base.hazard_csi,
        }

    def _score(self, workspace: Path, split: str) -> MetricsBundle:
        workspace = Path(workspace)
        timeout = int(self.config.research.experiment_timeout_sec)
        train = self.adapter.load_split("train")
        extra_frames = []
        for source_id in self._extra_source_ids(workspace, timeout):
            raw = self.adapter.extra_frame(source_id)
            admitted, decision = self.adapter.filter_external(source_id, raw)
            self.leakage_log.append(_decision_dict(decision))
            if admitted is not None:
                extra_frames.append(admitted)
        if extra_frames:
            train = self.adapter.concat([train, *extra_frames])
        eval_frame = self.adapter.load_split(split)
        y_hat = run_pipeline_subprocess(
            workspace, train, eval_frame, timeout_sec=timeout
        )
        bundle = self.adapter.score(y_hat, eval_frame, split=split)
        if self.leakage_log:
            bundle.leakage = list(self.leakage_log[-8:])
        return bundle

    def _extra_source_ids(self, workspace: Path, timeout: int) -> list[str]:
        """Union of manifest-declared sources (JSON, parsed here) and probed data.py ids."""
        ids = probe_extra_source_ids(workspace, timeout_sec=timeout)
        manifest = workspace / "external_manifest.json"
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise EvaluatorError(f"external_manifest.json malformed: {exc}") from exc
            for item in payload.get("sources") or payload.get("source_ids") or []:
                if str(item) not in ids:
                    ids.append(str(item))
        return ids


def _decision_dict(decision: Any) -> dict[str, Any]:
    if hasattr(decision, "to_dict"):
        return decision.to_dict()
    return {
        "source_id": getattr(decision, "source_id", ""),
        "admitted": bool(getattr(decision, "admitted", False)),
        "reason": getattr(decision, "reason", ""),
        "overlap_rate": float(getattr(decision, "overlap_rate", 0.0)),
        "removed_rows": int(getattr(decision, "removed_rows", 0)),
        "kept_rows": int(getattr(decision, "kept_rows", 0)),
        "layers": list(getattr(decision, "layers", [])),
    }
