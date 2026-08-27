"""Prepare an isolated trial workspace, apply the axis lock, then score val."""

from __future__ import annotations

import shutil
from pathlib import Path

from clh.core.errors import AxisLockError
from clh.research.axis_lock import apply_action, assert_axis_edits, restore_non_axis_files
from clh.research.cards import ActionCard, MetricsBundle
from clh.research.evaluator import IndependentEvaluator


def run_trial(
    *,
    pristine: Path,
    trial_dir: Path,
    action: ActionCard,
    evaluator: IndependentEvaluator,
) -> tuple[MetricsBundle, list[str]]:
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    shutil.copytree(pristine, trial_dir)
    restore_non_axis_files(trial_dir, pristine, action.axis)
    apply_action(trial_dir, action)
    changed = assert_axis_edits(pristine, trial_dir, action.axis)
    if not changed:
        raise AxisLockError("trial submitted no axis-local edits")
    metrics = evaluator.evaluate_workspace(trial_dir, split="val")
    return metrics, changed
