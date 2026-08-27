"""P0 custody tests: trial code runs isolated and cannot see eval labels or secrets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from clh.config import HarnessConfig
from clh.core.errors import EvaluatorError
from clh.domain.dummy.weather import build_weather, load_split
from clh.research.axis_lock import allowed_files
from clh.research.presets import preset_files
from clh.research.subproc import dump_frame, load_frame, run_pipeline_subprocess

DUMMY_PIPELINE = Path(__file__).resolve().parents[1] / "examples" / "dummy_research" / "pipeline"


@pytest.fixture(scope="module")
def frames():
    weather = build_weather()
    return load_split(weather, "train"), load_split(weather, "val")


def test_frame_roundtrip(tmp_path, frames):
    train, _ = frames
    dump_frame(train, tmp_path / "f.npz", strip_labels=False)
    loaded = load_frame(tmp_path / "f.npz")
    assert np.allclose(loaded.y, train.y)
    assert loaded.hazard_threshold == train.hazard_threshold
    assert list(loaded.airports[:3]) == [str(a) for a in train.airports[:3]]


def test_eval_labels_stripped(tmp_path, frames):
    _, val = frames
    dump_frame(val, tmp_path / "e.npz", strip_labels=True)
    loaded = load_frame(tmp_path / "e.npz")
    assert not hasattr(loaded, "y")
    assert not hasattr(loaded, "hazard")
    assert hasattr(loaded, "wind_speed")  # legitimate input feature stays


def test_baseline_pipeline_runs_in_subprocess(frames):
    train, val = frames
    y_hat = run_pipeline_subprocess(DUMMY_PIPELINE, train, val, timeout_sec=120)
    assert y_hat.shape == val.y.shape
    assert np.isfinite(y_hat).all()


def test_label_echo_cheat_fails(tmp_path, frames):
    """A trial that returns eval_frame.y must crash: labels are not shipped."""
    train, val = frames
    ws = tmp_path / "cheat"
    ws.mkdir()
    for name in ("features.py", "model.py", "objective.py", "physics.py", "data.py"):
        ws.joinpath(name).write_text((DUMMY_PIPELINE / name).read_text(), encoding="utf-8")
    ws.joinpath("pipeline.py").write_text(
        "def fit_predict(train_frame, eval_frame):\n    return eval_frame.y\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluatorError):
        run_pipeline_subprocess(ws, train, val, timeout_sec=120)


def test_subprocess_env_has_no_secrets(tmp_path, frames):
    train, val = frames
    os.environ["CLH_API_KEY"] = "sk-should-never-leak"
    try:
        ws = tmp_path / "env"
        ws.mkdir()
        ws.joinpath("pipeline.py").write_text(
            "import os\n"
            "def fit_predict(train_frame, eval_frame):\n"
            "    assert 'CLH_API_KEY' not in os.environ, 'secret leaked into trial env'\n"
            "    return eval_frame.wind_speed\n",
            encoding="utf-8",
        )
        y_hat = run_pipeline_subprocess(ws, train, val, timeout_sec=120)
        assert y_hat.shape == val.y.shape
    finally:
        os.environ.pop("CLH_API_KEY", None)


def test_objective_axis_registered():
    config = HarnessConfig.model_validate(
        {"research": {"axes": ["representation", "objective"]}}
    )
    assert config.research.axes == ["representation", "objective"]
    assert allowed_files("objective") == frozenset({"objective.py"})
    files = preset_files("objective", "extreme_wind_weights", DUMMY_PIPELINE, domain="aerowf")
    assert set(files) == {"objective.py"}
