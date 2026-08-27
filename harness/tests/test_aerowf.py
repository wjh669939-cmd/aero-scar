from pathlib import Path

import pytest

from clh.config import load_config
from clh.core.errors import EvaluatorError
from clh.domain.aerowf.io import assert_search_path_allowed
from clh.domain.aerowf.leakage import filter_external_source
from clh.domain.aerowf.world import build_aerowf_world
from clh.domain.protocol import load_adapter
from clh.plugins.compose import boot_context
from clh.research.cards import ActionCard
from clh.research.evaluator import IndependentEvaluator
from clh.research.experiment import run_trial
from clh.research.loop import ClosedLoopResearcher
from clh.research.presets import preset_files
from clh.research.reward import classify_signature

ROOT = Path(__file__).resolve().parents[1]
AEROWF_PIPELINE = ROOT / "examples" / "aerowf_research" / "pipeline"
DATA_ROOT = (
    ROOT.parent / "AeroWF" / "数据接口" / "AeroWF_v1_MODEL_TRAINING" / "release_v1"
)


@pytest.fixture
def aerowf_config(tmp_path: Path):
    if not DATA_ROOT.is_dir():
        pytest.skip("AeroWF release_v1 is not mounted")
    config = load_config(ROOT / "configs" / "atc.toml")
    config.llm.provider = "offline"
    config.research.budget_per_axis = 1
    config.domain.max_samples_per_split = 24
    config.domain.data_root = str(DATA_ROOT)
    return config


def test_search_path_guard_rejects_sealed() -> None:
    with pytest.raises(EvaluatorError):
        assert_search_path_allowed(Path("release_v1/sealed/temporal/ZBAA"))
    with pytest.raises(EvaluatorError):
        assert_search_path_allowed(Path("release_v1/pretrain/test/ZBAA"))
    with pytest.raises(EvaluatorError):
        assert_search_path_allowed(Path("release_v1/trainval/train/ZBAD"))
    assert_search_path_allowed(Path("release_v1/trainval/train/ZBAA"))
    assert_search_path_allowed(Path("release_v1/pretrain/train/ZBAA"))


def test_same_source_reject_and_pretrain_admit(aerowf_config, tmp_path: Path) -> None:
    world = build_aerowf_world(aerowf_config, ROOT)
    admitted, decision = filter_external_source(world.frames, "leak_val", world.frames["leak_val"])
    assert admitted is None
    assert decision.admitted is False
    assert "same-source" in decision.reason
    if "pretrain_train" in world.frames:
        extra, extra_decision = filter_external_source(
            world.frames, "pretrain_train", world.frames["pretrain_train"]
        )
        assert extra_decision.admitted is True
        assert extra is not None
        assert len(extra) > 0


def test_ridge_beats_persistence_on_aerowf_val(aerowf_config, tmp_path: Path) -> None:
    adapter = load_adapter(aerowf_config, ROOT)
    evaluator = IndependentEvaluator(aerowf_config, adapter, AEROWF_PIPELINE)
    action = ActionCard(
        axis="model",
        kind="preset",
        preset="ridge",
        files=preset_files("model", "ridge", AEROWF_PIPELINE, domain="aerowf"),
    )
    metrics, _ = run_trial(
        pristine=AEROWF_PIPELINE,
        trial_dir=tmp_path / "ridge_trial",
        action=action,
        evaluator=evaluator,
    )
    assert metrics.mae < evaluator.baseline.mae
    assert "overall.mae" in metrics.endpoints
    assert set(adapter.world.source_airports) <= set(metrics.per_airport_mae)


def test_search_cannot_score_aerowf_test(aerowf_config) -> None:
    adapter = load_adapter(aerowf_config, ROOT)
    evaluator = IndependentEvaluator(aerowf_config, adapter, AEROWF_PIPELINE)
    with pytest.raises(EvaluatorError):
        evaluator.evaluate_workspace(AEROWF_PIPELINE, split="test_temporal")  # type: ignore[arg-type]


def test_offline_aerowf_closed_loop(aerowf_config, tmp_path: Path) -> None:
    ctx = boot_context(aerowf_config, tmp_path / "run", workspace_root=ROOT)
    researcher = ClosedLoopResearcher(ctx)
    summary = researcher.run()
    # G3 冻结轴配置（atc.toml）：representation + objective 双轴，budget_per_axis=1
    assert summary["n_trials"] == 2
    cert = summary["certification"]
    assert cert["protocol"].startswith("arXiv:2606.22731")
    assert "routed" in cert
    assert "test_temporal" in cert["routed"]
    # 结构性断言：候选可能为空（小样本下 preset 未过阈值是合法结果），
    # 但凡有候选，其轴必须落在冻结的双轴配置内
    assert isinstance(cert["axes"], dict)
    assert set(cert["axes"]) <= {"representation", "objective"}
    session = (tmp_path / "run" / "session.jsonl").read_text(encoding="utf-8")
    assert "trial_result" in session
    assert "test_labels" not in session
    visible = ctx.get("session").model_visible()
    assert all(row["type"] != "certification" for row in visible)


def test_signature_names() -> None:
    assert classify_signature(0.041, 0.003) == "selection_variance"
    assert classify_signature(0.022, -0.019) == "distribution_shift"
    assert classify_signature(0.032, 0.013) == "transfer"
