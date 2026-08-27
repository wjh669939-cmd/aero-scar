"""A-2: 文件级 axis lock 匹配引擎。

输入: 改动文件清单 + 轴名 + axis_lock 配置（JSON）
输出: LockDecision（allowed / violations），供执行器在训练前拒绝越界 patch。

配置里的 <AEROWF_REPO> / <DOWNSTREAM> 占位符由调用方传入两个根路径替换；
匹配在绝对 POSIX 路径空间进行（AeroWF 仓库与 downstream 工作区双根并存，
相对化到单根会产生歧义）。
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


@dataclass
class LockDecision:
    axis: str
    allowed: bool
    changed: list[str]
    violations: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "allowed": self.allowed,
            "changed": self.changed,
            "violations": self.violations,
            "reason": self.reason,
        }


def _abs_posix(path: str, default_root: str) -> str:
    p = PurePosixPath(str(path).replace("\\", "/"))
    if not p.is_absolute():
        p = PurePosixPath(str(default_root).replace("\\", "/")) / p
    return str(p)


def _matches(abs_path: str, pattern: str) -> bool:
    """fnmatch 变体：'dir/**' 匹配 dir 下任意深度。"""
    pattern = pattern.replace("\\", "/")
    if pattern.endswith("/**"):
        base = pattern[:-3]
        return abs_path == base or abs_path.startswith(base + "/")
    return fnmatch.fnmatch(abs_path, pattern)


def load_config(
    config_path: Path,
    repo_root: str,
    downstream_root: str | None = None,
) -> dict:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ds_root = downstream_root if downstream_root is not None else repo_root

    def substitute(pattern: str) -> str:
        return (
            pattern.replace("<AEROWF_REPO>", repo_root.rstrip("/"))
            .replace("<DOWNSTREAM>", ds_root.rstrip("/"))
        )

    axes = {}
    for axis, spec in raw["axis_paths"].items():
        axes[axis] = {
            "allow": [substitute(p) for p in spec["allow"]],
            "active": bool(spec.get("active", True)),
        }
    forbidden_tokens = [
        t
        for t in raw.get("hidden_data_guard", {}).get("certification_paths_forbidden_tokens", [])
        if not (t.startswith("<") and t.endswith(">"))
    ]
    return {
        "axes": axes,
        "repo_root": repo_root,
        "downstream_root": ds_root,
        "forbidden_tokens": forbidden_tokens,
    }


def check(changed_paths: list[str], axis: str, config: dict) -> LockDecision:
    changed = [_abs_posix(p, config["repo_root"]) for p in changed_paths]
    spec = config["axes"].get(axis)
    if spec is None:
        return LockDecision(axis, False, changed, changed, f"unknown axis: {axis}")
    if not spec["active"]:
        return LockDecision(axis, False, changed, changed, f"axis not active: {axis}")
    if not changed:
        return LockDecision(axis, False, changed, [], "empty diff: trial submitted no edits")
    token_hits = [
        p for p in changed
        if any(tok in p for tok in config.get("forbidden_tokens", []))
    ]
    if token_hits:
        return LockDecision(
            axis, False, changed, token_hits,
            "hidden-data path token matched: certification paths are evaluator-only",
        )
    violations = [
        p for p in changed if not any(_matches(p, pat) for pat in spec["allow"])
    ]
    if violations:
        return LockDecision(
            axis, False, changed, violations,
            f"axis violation: {len(violations)} file(s) outside allowed paths for {axis}",
        )
    return LockDecision(axis, True, changed)
