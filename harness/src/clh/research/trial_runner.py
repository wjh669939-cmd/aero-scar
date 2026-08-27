"""Entry point executed INSIDE the isolated trial subprocess.

Runs with a whitelisted environment (see subproc._clean_env): no API keys,
no sealed paths, workspace as cwd. Two modes:

  --mode extras : import workspace data.py, write its extra_source_ids() list
  --mode fit    : load serialized frames, import workspace pipeline.py,
                  run fit_predict, save predictions.npy

This module must stay dependency-light; it only imports numpy and
clh.research.subproc (for frame deserialization).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mode_extras(workspace: Path, out: Path) -> None:
    data_py = workspace / "data.py"
    ids: list[str] = []
    if data_py.is_file():
        module = _import_file("clh_trial_data", data_py)
        fn = getattr(module, "extra_source_ids", None)
        if callable(fn):
            ids = [str(i) for i in fn()]
    out.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")


def _mode_fit(workspace: Path, train_path: Path, eval_path: Path, out: Path) -> None:
    from clh.research.subproc import load_frame

    pipeline_py = workspace / "pipeline.py"
    if not pipeline_py.is_file():
        raise RuntimeError("pipeline.py is missing from the trial workspace")
    # trial modules (features/model/...) resolve against the workspace only
    sys.path.insert(0, str(workspace))
    train = load_frame(train_path)
    eval_frame = load_frame(eval_path)
    module = _import_file("clh_trial_pipeline", pipeline_py)
    fit_predict = getattr(module, "fit_predict", None)
    if fit_predict is None:
        raise RuntimeError("pipeline.py must export fit_predict(train_frame, eval_frame)")
    y_hat = np.asarray(fit_predict(train, eval_frame), dtype=float)
    np.save(out, y_hat)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extras", "fit"], required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--train", default="")
    parser.add_argument("--eval", dest="eval_path", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    if args.mode == "extras":
        _mode_extras(workspace, Path(args.out))
    else:
        _mode_fit(workspace, Path(args.train), Path(args.eval_path), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
