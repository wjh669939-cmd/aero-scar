"""Deterministic specialist used when no model API key is configured."""

from __future__ import annotations

import json

from clh.config import AxisName
from clh.llm.provider import LLMResponse, LLMUsage
from clh.llm.openai_compat import extract_json_object


class OfflineLLM:
    """Scripted researcher. Lets the loop run tests and demos without an API."""

    name = "offline"

    def __init__(self) -> None:
        self._calls = 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        label: str = "",
        json_mode: bool = False,
    ) -> LLMResponse:
        del json_mode
        payload = self._payload(system, user, label)
        text = json.dumps(payload, ensure_ascii=False)
        usage = LLMUsage(model="offline", prompt_tokens=0, completion_tokens=0, total_tokens=0, source="offline")
        return LLMResponse(text=text, usage=usage)

    def ask_json(self, system: str, user: str, *, label: str = "") -> dict:
        return extract_json_object(self.complete(system, user, label=label).text)

    def _payload(self, system: str, user: str, label: str) -> dict:
        axis = _axis_from_text(user) or _axis_from_text(system)
        self._calls += 1
        kind = (label or "").lower()
        if kind == "action" or kind.startswith("action"):
            return _action(axis)
        if kind == "hypothesis" or kind.startswith("hypothesis"):
            return _hypothesis(axis, self._calls)
        blob = f"{system}\n{user}".lower()
        if "return json: {\"axis\"" in blob or "kind\": \"preset\"" in blob or "convert an approved hypothesis" in blob:
            return _action(axis)
        return _hypothesis(axis, self._calls)


def _axis_from_text(text: str) -> AxisName:
    import re

    match = re.search(r"assigned axis:\s*(representation|model|physics|objective|data)", text, flags=re.I)
    if match:
        return match.group(1).lower()  # type: ignore[return-value]
    match = re.search(r"^axis:\s*(representation|model|physics|objective|data)", text, flags=re.I | re.M)
    if match:
        return match.group(1).lower()  # type: ignore[return-value]
    lowered = text.lower()
    for name in ("representation", "model", "physics", "objective", "data"):
        if f"axis: {name}" in lowered or f"axis={name}" in lowered:
            return name  # type: ignore[return-value]
    return "model"


def _hypothesis(axis: AxisName, step: int) -> dict:
    catalog = {
        "representation": {
            "claim": "Runway-relative wind components explain short-horizon wind better than raw speed.",
            "mechanism": "Headwind and crosswind are the operational quantities; persistence on scalar speed ignores runway geometry.",
            "target_slice": "T+1 wind MAE on source airports",
            "expected_gain": "Validation MAE should drop versus persistence.",
            "side_effects": "Hazard CSI should not drop.",
            "falsification": "If MAE does not improve on at least two source airports, abandon.",
            "negative_control": "Hour-of-day encoding alone should not match the gain.",
        },
        "model": {
            "claim": "A linear estimator on physics-aware features outperforms persistence.",
            "mechanism": "The synthetic process is approximately linear in lagged wind plus headwind.",
            "target_slice": "overall MAE",
            "expected_gain": "Positive normalised MAE improvement.",
            "side_effects": "Must keep high-wind CSI within safety tolerance.",
            "falsification": "If CSI falls more than the safety gate, reject.",
            "negative_control": "Shuffled-label fit must not improve MAE.",
        },
        "physics": {
            "claim": "Upweighting high-wind samples reduces operational hazard misses.",
            "mechanism": "Uniform MAE underfits rare strong-wind events.",
            "target_slice": "high-wind CSI",
            "expected_gain": "CSI up, MAE not worse by more than 5%.",
            "side_effects": "Average MAE may rise slightly.",
            "falsification": "If CSI does not rise, the weight is not the mechanism.",
            "negative_control": "Uniform weights should not raise CSI.",
        },
        "objective": {
            "claim": "Upweighting high-wind samples in the loss reduces operational hazard misses.",
            "mechanism": "Uniform MAE underfits rare strong-wind events; loss reweighting shifts capacity to the hazardous tail.",
            "target_slice": "high-wind CSI",
            "expected_gain": "CSI up, MAE not worse by more than 5%.",
            "side_effects": "Average MAE may rise slightly.",
            "falsification": "If CSI does not rise, the weight is not the mechanism.",
            "negative_control": "Uniform weights should not raise CSI.",
        },
        "data": {
            "claim": "Extra labelled hours from a climate-matched airport help; a shifted climate does not.",
            "mechanism": "Matched climatology expands coverage without changing the forecast mapping.",
            "target_slice": "source-airport MAE",
            "expected_gain": "Small MAE gain on validation.",
            "side_effects": "Spatial certification may reverse if the extra airport is a different climate.",
            "falsification": "If the leakage filter rejects the source, do not merge.",
            "negative_control": "Same-source overlap must be rejected.",
        },
    }
    card = catalog[axis]
    return {
        "axis": axis,
        "trial_index": step,
        **card,
    }


def _action(axis: AxisName) -> dict:
    return {
        "axis": axis,
        "kind": "preset",
        "preset": {
            "representation": "runway_wind",
            "model": "ridge",
            "physics": "extreme_wind_weights",
            "objective": "extreme_wind_weights",
            "data": "matched_airport",
        }[axis],
    }
