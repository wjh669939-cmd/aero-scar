"""Run configuration. Profiles compose like dsh bundles, using TOML/YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import yaml

AxisName = Literal["data", "representation", "model", "physics", "objective"]
CertSplit = Literal["temporal", "spatial", "event"]


class LLMConfig(BaseModel):
    """OpenAI-compatible chat provider settings."""

    provider: Literal["openai_compat", "offline"] = "offline"
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    temperature: float = 0.2
    timeout_sec: float = 60.0
    max_output_tokens: int = 4096
    retry_attempts: int = 3
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 12.0

    @field_validator("api_key", "base_url", "model", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ResearchConfig(BaseModel):
    """Closed-loop search budget and selection policy."""

    budget_per_axis: int = Field(default=3, ge=1)
    axes: list[AxisName] = Field(
        default_factory=lambda: ["representation", "model", "physics", "data"]
    )
    equal_budget: bool = True
    selection_threshold: float = 0.005
    safety_csi_tolerance: float = 0.02
    experiment_timeout_sec: int = Field(default=60, ge=5)
    smoke_max_samples: int = 64


class DomainConfig(BaseModel):
    """Which scientific task the harness is attached to."""

    name: str = "dummy"
    baseline: str = "persistence"
    pipeline_root: str = "examples/dummy_research/pipeline"
    data_root: str = ""
    sealed_root: str = ""
    source_airports: list[str] = Field(default_factory=lambda: ["ZBAA", "ZSPD", "ZSSS"])
    holdout_airports: list[str] = Field(default_factory=lambda: ["ZBAD"])
    max_samples_per_split: int = Field(default=0, ge=0)
    inner_val_frac: float = Field(default=0.20, ge=0.05, le=0.5)
    inner_split_seed: int = 42


class CertificationConfig(BaseModel):
    """One-shot held-out protocol. Test labels never enter the search loop."""

    splits: list[CertSplit] = Field(default_factory=lambda: ["temporal", "spatial", "event"])
    freeze_top_k_per_axis: int = 1
    routing_threshold: float = 0.005
    sealed_required: bool = False


class HarnessConfig(BaseModel):
    """Root config loaded from TOML/YAML plus environment overrides."""

    profile: str = "dummy"
    run_root: str = "runs"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    certification: CertificationConfig = Field(default_factory=CertificationConfig)

    def resolved_pipeline_root(self, workspace: Path) -> Path:
        return _resolve_path(workspace, self.domain.pipeline_root)

    def resolved_data_root(self, workspace: Path) -> Path | None:
        if not self.domain.data_root:
            return None
        return _resolve_path(workspace, self.domain.data_root)

    def resolved_sealed_root(self, workspace: Path) -> Path | None:
        if not self.domain.sealed_root:
            return None
        return _resolve_path(workspace, self.domain.sealed_root)


def _resolve_path(workspace: Path, raw: str) -> Path:
    root = Path(raw)
    if root.is_absolute():
        return root
    return (workspace / root).resolve()


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = tomllib.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must be a mapping")
    return loaded


def _env_override(data: dict[str, Any]) -> dict[str, Any]:
    llm = dict(data.get("llm") or {})
    llm["api_key"] = os.environ.get("CLH_API_KEY") or os.environ.get("OPENAI_API_KEY") or llm.get("api_key", "")
    llm["base_url"] = os.environ.get("CLH_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or llm.get("base_url", "")
    llm["model"] = os.environ.get("CLH_MODEL") or os.environ.get("SIMPLE_AR_MODEL") or llm.get("model", "deepseek-chat")
    if os.environ.get("CLH_LLM_TIMEOUT_SEC"):
        llm["timeout_sec"] = float(os.environ["CLH_LLM_TIMEOUT_SEC"])
    if os.environ.get("CLH_MAX_OUTPUT_TOKENS"):
        llm["max_output_tokens"] = int(os.environ["CLH_MAX_OUTPUT_TOKENS"])
    data["llm"] = llm
    return data


def load_config(path: str | Path | None = None) -> HarnessConfig:
    """Load a profile file, then overlay CLH_* / OPENAI_* environment variables."""
    from dotenv import load_dotenv

    load_dotenv()
    data: dict[str, Any] = {}
    if path is not None:
        data = _read_mapping(Path(path))
    return HarnessConfig.model_validate(_env_override(data))
