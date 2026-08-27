from clh.config import load_config
from clh.core.session import SessionLog
from clh.llm.openai_compat import extract_json_object
from clh.research.reward import normalised_improvement


def test_extract_json_from_fence() -> None:
    text = "```json\n{\"axis\": \"model\", \"claim\": \"x\"}\n```"
    assert extract_json_object(text)["axis"] == "model"


def test_normalised_improvement_mae() -> None:
    gain = normalised_improvement(0.8, 1.0, higher_is_better=False)
    assert abs(gain - 0.2) < 1e-9
    up = normalised_improvement(0.9, 0.8, higher_is_better=True)
    assert abs(up - 0.125) < 1e-9


def test_session_hides_certification_scores(tmp_path) -> None:
    log = SessionLog(tmp_path / "session.jsonl")
    log.append("trial_result", improvement=0.1)
    log.append("certification_score", mae=0.0)
    visible = log.model_visible()
    assert all(row["type"] != "certification_score" for row in visible)


def test_load_config_env(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text('profile = "x"\n[llm]\nprovider = "offline"\n', encoding="utf-8")
    monkeypatch.setenv("CLH_MODEL", "deepseek-chat")
    monkeypatch.setenv("CLH_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("CLH_API_KEY", "sk-test")
    config = load_config(cfg_path)
    assert config.llm.model == "deepseek-chat"
    assert config.llm.api_key == "sk-test"
    assert "deepseek" in config.llm.base_url
