"""P0 custody fix: run agent-authored trial code in an isolated subprocess.

Before this module, ``evaluator.py`` imported the trial's ``pipeline.py`` into
the evaluator's own process, so agent code ran with the same privileges as the
component that will later hold sealed-data access. Now:

  evaluator process              trial subprocess (clean env)
  ---------------------          -----------------------------
  serialize train/eval  ------>  load frames (labels stripped on eval)
  frames to npz                  import workspace pipeline.py
                                 fit_predict -> predictions.npy
  score predictions     <------  exit

The eval frame handed to the subprocess has ``y`` / ``hazard`` /
``weather_label`` removed, closing a second hole: in-process trials could read
validation labels at predict time and echo them back.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from clh.core.errors import EvaluatorError

#: target/label attributes never shown to trial code on the eval split
LABEL_FIELDS = ("y", "hazard", "weather_label")

_META_SUFFIX = ".meta.json"


def dump_frame(frame: Any, npz_path: Path, *, strip_labels: bool = False) -> None:
    """Serialize a dataclass-style frame (AeroFrame / WeatherFrame) to npz + meta json."""
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"_object_keys": []}
    for key, value in vars(frame).items():
        if value is None:
            continue
        if strip_labels and key in LABEL_FIELDS:
            continue
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                arrays[key] = value.astype("U")
                meta["_object_keys"].append(key)
            else:
                arrays[key] = value
        elif isinstance(value, (int, float, str, bool)):
            meta[key] = value
    np.savez(npz_path, **arrays)
    Path(str(npz_path) + _META_SUFFIX).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )


def load_frame(npz_path: Path) -> SimpleNamespace:
    """Rebuild a frame as attribute-access namespace. Methods are not restored."""
    meta_path = Path(str(npz_path) + _META_SUFFIX)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    payload: dict[str, Any] = {k: v for k, v in meta.items() if not k.startswith("_")}
    with np.load(npz_path, allow_pickle=False) as data:
        for key in data.files:
            payload[key] = data[key]
    return SimpleNamespace(**payload)


def _clean_env() -> dict[str, str]:
    """Whitelist env for the trial subprocess: no API keys, no sealed paths."""
    src_root = str(Path(__file__).resolve().parents[2])
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "PYTHONPATH": src_root,
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
    }


def _run(args: list[str], *, cwd: Path, timeout_sec: int, what: str) -> None:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "clh.research.trial_runner", *args],
            cwd=cwd,
            env=_clean_env(),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvaluatorError(f"{what} timed out after {timeout_sec}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise EvaluatorError(f"{what} failed (exit {proc.returncode}): {tail}")


def probe_extra_source_ids(workspace: Path, *, timeout_sec: int = 60) -> list[str]:
    """Phase 1 (data axis): call the workspace's data.py in isolation, if present."""
    if not (workspace / "data.py").is_file():
        return []
    with tempfile.TemporaryDirectory(prefix="clh-probe-") as tmp:
        out = Path(tmp) / "extra_ids.json"
        _run(
            ["--mode", "extras", "--workspace", str(workspace), "--out", str(out)],
            cwd=workspace,
            timeout_sec=timeout_sec,
            what="extras probe subprocess",
        )
        if not out.exists():
            return []
        ids = json.loads(out.read_text(encoding="utf-8"))
        return [str(i) for i in ids] if isinstance(ids, list) else []


def run_pipeline_subprocess(
    workspace: Path,
    train_frame: Any,
    eval_frame: Any,
    *,
    timeout_sec: int,
) -> np.ndarray:
    """Phase 2: fit_predict in isolation; returns predictions as float ndarray."""
    with tempfile.TemporaryDirectory(prefix="clh-trial-") as tmp:
        io_dir = Path(tmp)
        dump_frame(train_frame, io_dir / "train.npz", strip_labels=False)
        dump_frame(eval_frame, io_dir / "eval.npz", strip_labels=True)
        out = io_dir / "predictions.npy"
        _run(
            [
                "--mode", "fit",
                "--workspace", str(workspace),
                "--train", str(io_dir / "train.npz"),
                "--eval", str(io_dir / "eval.npz"),
                "--out", str(out),
            ],
            cwd=workspace,
            timeout_sec=timeout_sec,
            what="trial subprocess",
        )
        if not out.exists():
            raise EvaluatorError("trial subprocess produced no predictions.npy")
        y_hat = np.load(out, allow_pickle=False)
    return np.asarray(y_hat, dtype=float)
