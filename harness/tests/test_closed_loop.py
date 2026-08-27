from pathlib import Path

import pytest

from clh.config import load_config
from clh.core.errors import AxisLockError, EvaluatorError
from clh.domain.atc.leakage import filter_external_source
from clh.domain.dummy.weather import build_weather
from clh.plugins.compose import boot_context
from clh.research.axis_lock import apply_action, assert_axis_edits
from clh.research.cards import ActionCard
from clh.research.evaluator import IndependentEvaluator
from clh.research.loop import ClosedLoopResearcher
from clh.research.presets import preset_files

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "examples" / "dummy_research" / "pipeline"


@pytest.fixture
def weather():
    return build_weather()


def test_axis_lock_rejects_cross_axis_edit(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    import shutil

    shutil.copytree(PIPELINE, workspace)
    (workspace / "model.py").write_text("# hacked\n", encoding="utf-8")
    with pytest.raises(AxisLockError):
        assert_axis_edits(PIPELINE, workspace, "representation")


def test_axis_lock_allows_model_only(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    import shutil

    shutil.copytree(PIPELINE, workspace)
    action = ActionCard(
        axis="model",
        kind="preset",
        preset="ridge",
        files=preset_files("model", "ridge", PIPELINE),
    )
    apply_action(workspace, action)
    changed = assert_axis_edits(PIPELINE, workspace, "model")
    assert "model.py" in changed


def test_leakage_rejects_same_source(weather) -> None:
    frame = weather.frames["leak_ZBAA_future"]
    admitted, decision = filter_external_source(weather, "leak_ZBAA_future", frame)
    assert admitted is None
    assert decision.admitted is False
    assert "same-source" in decision.reason


def test_leakage_admits_matched_airport(weather) -> None:
    frame = weather.frames["matched_ZBHH"]
    admitted, decision = filter_external_source(weather, "matched_ZBHH", frame)
    assert decision.admitted is True
    assert admitted is not None
    assert len(admitted) > 0


def test_search_cannot_score_test_split(weather) -> None:
    evaluator = IndependentEvaluator(load_config(ROOT / "configs" / "dummy.toml"), weather, PIPELINE)
    with pytest.raises(EvaluatorError):
        evaluator.evaluate_workspace(PIPELINE, split="test_temporal")  # type: ignore[arg-type]


def test_ridge_beats_persistence_on_val(weather, tmp_path: Path) -> None:
    from clh.research.experiment import run_trial

    config = load_config(ROOT / "configs" / "dummy.toml")
    evaluator = IndependentEvaluator(config, weather, PIPELINE)
    action = ActionCard(
        axis="model",
        kind="preset",
        preset="ridge",
        files=preset_files("model", "ridge", PIPELINE),
    )
    metrics, _ = run_trial(
        pristine=PIPELINE,
        trial_dir=tmp_path / "ridge_trial",
        action=action,
        evaluator=evaluator,
    )
    assert metrics.mae < evaluator.baseline.mae


def test_offline_closed_loop(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "dummy.toml")
    ctx = boot_context(config, tmp_path / "run", workspace_root=ROOT)
    researcher = ClosedLoopResearcher(ctx)
    summary = researcher.run()
    assert summary["n_trials"] == 4
    by_axis = {trial.axis: trial for trial in researcher.trials}
    assert by_axis["model"].status == "improved"
    assert by_axis["model"].improvement > 0
    cert = summary["certification"]
    assert "test_temporal" in cert["baseline"]
    assert "model" in cert["axes"]
    assert "routed" in cert
    session = (tmp_path / "run" / "session.jsonl").read_text(encoding="utf-8")
    assert "trial_result" in session
    assert "test_labels" not in session

