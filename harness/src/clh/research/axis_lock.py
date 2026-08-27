"""File-level ablation lock. One trial may edit only its assigned axis."""

from __future__ import annotations

import shutil
from pathlib import Path

from clh.config import AxisName
from clh.core.errors import AxisLockError
from clh.research.cards import ActionCard

AXIS_FILES: dict[AxisName, frozenset[str]] = {
    "data": frozenset({"data.py", "external_manifest.json"}),
    "representation": frozenset({"features.py"}),
    "model": frozenset({"model.py"}),
    "physics": frozenset({"physics.py", "objective.py"}),
    # aerowf-v1 正式轴（05 文档）：objective 轴独立于 physics；
    # atc profile 使用 ["representation", "objective"]，physics/data 保留给 dummy 协议测试。
    "objective": frozenset({"objective.py"}),
}

LOCKED_ALWAYS = frozenset({"README.md"})


def allowed_files(axis: AxisName) -> frozenset[str]:
    return AXIS_FILES[axis]


def restore_non_axis_files(workspace: Path, pristine: Path, axis: AxisName) -> list[str]:
    """Copy every file outside the active axis from the pristine tree."""
    restored: list[str] = []
    allow = allowed_files(axis)
    for source in pristine.rglob("*"):
        if not source.is_file():
            continue
        rel = source.relative_to(pristine).as_posix()
        if rel in allow:
            continue
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        restored.append(rel)
    return restored


def assert_axis_edits(pristine: Path, workspace: Path, axis: AxisName) -> list[str]:
    """Reject a trial whose diff touches files outside the active axis."""
    allow = allowed_files(axis)
    changed: list[str] = []
    violations: list[str] = []
    workspace_files = {path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()}
    pristine_files = {path.relative_to(pristine).as_posix() for path in pristine.rglob("*") if path.is_file()}
    for rel in sorted(workspace_files | pristine_files):
        left = pristine / rel
        right = workspace / rel
        if not left.exists() or not right.exists():
            if rel in allow:
                changed.append(rel)
            else:
                violations.append(rel)
            continue
        if left.read_bytes() != right.read_bytes():
            if rel in allow:
                changed.append(rel)
            else:
                violations.append(rel)
    if violations:
        raise AxisLockError(
            f"axis {axis} may only edit {sorted(allow)}; also changed {violations}"
        )
    return changed


def apply_action(workspace: Path, action: ActionCard) -> list[str]:
    """Write declared files. Callers must still run ``assert_axis_edits``."""
    written: list[str] = []
    allow = allowed_files(action.axis)
    for rel, content in action.files.items():
        if rel not in allow:
            raise AxisLockError(f"{rel} is not editable on axis {action.axis}")
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written.append(rel)
    return written
