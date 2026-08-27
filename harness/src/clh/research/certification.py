"""One-shot held-out certification after the search is frozen (paper §3.1–3.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clh.config import CertificationConfig
from clh.research.cards import MetricsBundle, TrialRecord
from clh.research.evaluator import IndependentEvaluator
from clh.research.reward import (
    aggregate_improvement,
    classify_signature,
    endpoint_improvements,
)


def select_best_per_axis(trials: list[TrialRecord], threshold: float) -> dict[str, TrialRecord]:
    """c_a = argmax mean_t I_t^val among retained snapshots (paper eq. 1)."""
    best: dict[str, TrialRecord] = {}
    for trial in trials:
        if trial.val_metrics is None or trial.status in {"failed", "rejected", "unsafe"}:
            continue
        current = best.get(trial.axis)
        if current is None or trial.improvement > current.improvement:
            best[trial.axis] = trial
    return {axis: trial for axis, trial in best.items() if trial.improvement >= threshold}


def certify_frozen(
    evaluator: IndependentEvaluator,
    best: dict[str, TrialRecord],
    trial_dirs: dict[str, str],
    config: CertificationConfig,
) -> dict[str, Any]:
    tau = config.routing_threshold
    report: dict[str, Any] = {
        "protocol": "arXiv:2606.22731 §3.1–3.3",
        "tau": tau,
        "axes": {},
        "baseline": {},
        "routed": {},
    }
    baseline_val = evaluator.baseline
    report["baseline"]["val"] = baseline_val.to_dict()
    test_baselines: dict[str, MetricsBundle] = {}
    for split in config.splits:
        name = f"test_{split}"
        bundle = evaluator.certify_workspace(evaluator.baseline_workspace, split=name)
        test_baselines[name] = bundle
        report["baseline"][name] = bundle.to_dict()

    axis_test: dict[str, dict[str, MetricsBundle]] = {}
    for axis, trial in best.items():
        workspace = Path(trial_dirs[trial.trial_id])
        val_metrics = trial.val_metrics or evaluator.baseline
        val_i = trial.improvement
        val_eps = endpoint_improvements(val_metrics, baseline_val)
        axis_row: dict[str, Any] = {
            "trial_id": trial.trial_id,
            "val": {
                "aggregate": val_i,
                "endpoints": val_eps,
                "metrics": val_metrics.to_dict(),
            },
            "splits": {},
        }
        axis_test[axis] = {}
        for split in config.splits:
            name = f"test_{split}"
            metrics = evaluator.certify_workspace(workspace, split=name)
            base = test_baselines[name]
            test_i = aggregate_improvement(metrics, base)
            test_eps = endpoint_improvements(metrics, base)
            axis_test[axis][name] = metrics
            axis_row["splits"][name] = {
                "metrics": metrics.to_dict(),
                "aggregate": test_i,
                "endpoints": test_eps,
                "improvement": test_i,
                "csi_gain": test_eps.get("hazard.csi", 0.0),
                "generalization_gap": val_i - test_i,
                "signature": classify_signature(val_i, test_i, tau=tau),
            }
        report["axes"][axis] = axis_row

    for split in config.splits:
        name = f"test_{split}"
        report["routed"][name] = _route_split(
            best, axis_test, baseline_val, test_baselines[name], name, tau
        )
    return report


def _route_split(
    best: dict[str, TrialRecord],
    axis_test: dict[str, dict[str, MetricsBundle]],
    val_baseline: MetricsBundle,
    test_baseline: MetricsBundle,
    split_name: str,
    tau: float,
) -> dict[str, Any]:
    """Per endpoint, pick the axis with highest val I_t >= tau, then read its held-out I_t."""
    common = [name for name in test_baseline.endpoints if name in val_baseline.endpoints]
    routed_eps: dict[str, Any] = {}
    heldout: list[float] = []
    for name in common:
        chosen_axis = None
        chosen_val = -1e9
        for axis, trial in best.items():
            if trial.val_metrics is None:
                continue
            val_eps = endpoint_improvements(trial.val_metrics, trial.baseline_metrics or val_baseline)
            val_i = val_eps.get(name)
            if val_i is None or val_i < tau:
                continue
            if val_i > chosen_val:
                chosen_val = val_i
                chosen_axis = axis
        if chosen_axis is None:
            routed_eps[name] = {"axis": None, "val": 0.0, "test": 0.0}
            continue
        test_metrics = axis_test[chosen_axis][split_name]
        test_eps = endpoint_improvements(test_metrics, test_baseline)
        test_i = float(test_eps.get(name, 0.0))
        heldout.append(test_i)
        routed_eps[name] = {
            "axis": chosen_axis,
            "val": chosen_val,
            "test": test_i,
            "signature": classify_signature(chosen_val, test_i, tau=tau),
        }
    aggregate = float(sum(heldout) / len(heldout)) if heldout else 0.0
    return {"aggregate": aggregate, "endpoints": routed_eps}
