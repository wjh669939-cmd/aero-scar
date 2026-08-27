"""Structured cards for falsification-driven Auto Research."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from clh.config import AxisName

TrialStatus = Literal["improved", "no_gain", "failed", "rejected", "unsafe"]


class HypothesisCard(BaseModel):
    """Pre-registered claim. Must exist before any pipeline edit."""

    axis: AxisName
    claim: str
    mechanism: str
    target_slice: str
    expected_gain: str
    side_effects: str = ""
    falsification: str
    negative_control: str = ""
    parent_trial_id: str | None = None

    def to_prompt(self) -> str:
        return (
            f"axis: {self.axis}\n"
            f"claim: {self.claim}\n"
            f"mechanism: {self.mechanism}\n"
            f"target_slice: {self.target_slice}\n"
            f"expected_gain: {self.expected_gain}\n"
            f"side_effects: {self.side_effects}\n"
            f"falsification: {self.falsification}\n"
            f"negative_control: {self.negative_control}"
        )


class ActionCard(BaseModel):
    """Single-axis intervention. Only files on this axis may change."""

    axis: AxisName
    kind: Literal["preset", "files"] = "preset"
    preset: str = ""
    files: dict[str, str] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("preset", "notes", mode="before")
    @classmethod
    def _empty_str(cls, value: object) -> object:
        return "" if value is None else value

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, value: object) -> object:
        return "preset" if value in {None, ""} else value

    @field_validator("files", mode="before")
    @classmethod
    def _files(cls, value: object) -> object:
        return {} if not value else value


class EndpointScore(BaseModel):
    """One endpoint t in the paper's set T (suite member)."""

    name: str
    value: float
    higher_is_better: bool
    kind: str = "mae"


class MetricsBundle(BaseModel):
    mae: float
    rmse: float
    hazard_csi: float
    per_airport_mae: dict[str, float] = Field(default_factory=dict)
    endpoints: dict[str, EndpointScore] = Field(default_factory=dict)
    split: str = "val"
    n_samples: int = 0
    leakage: list[dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EvidenceVerdict(BaseModel):
    supported: bool
    reason: str
    target_improved: bool
    negative_control_ok: bool
    safety_ok: bool
    status: TrialStatus


class TrialRecord(BaseModel):
    trial_id: str
    axis: AxisName
    parent_trial_id: str | None = None
    hypothesis: HypothesisCard
    action: ActionCard
    status: TrialStatus
    val_metrics: MetricsBundle | None = None
    baseline_metrics: MetricsBundle | None = None
    improvement: float = 0.0
    evidence: EvidenceVerdict | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
