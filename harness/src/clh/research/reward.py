"""Paper §3.2 eq. (3): per-endpoint normalised improvement and unweighted aggregate."""

from __future__ import annotations

from typing import Literal

from clh.research.cards import EndpointScore, MetricsBundle

Signature = Literal["transfer", "selection_variance", "distribution_shift", "none"]


def normalised_improvement(score: float, baseline: float, *, higher_is_better: bool) -> float:
    denom = abs(baseline) if abs(baseline) > 1e-8 else 1e-8
    if higher_is_better:
        return (score - baseline) / denom
    return (baseline - score) / denom


def endpoint_improvements(candidate: MetricsBundle, baseline: MetricsBundle) -> dict[str, float]:
    """I_t for every shared endpoint (paper eq. 3)."""
    out: dict[str, float] = {}
    for name, ep in candidate.endpoints.items():
        base = baseline.endpoints.get(name)
        if base is None:
            continue
        out[name] = normalised_improvement(ep.value, base.value, higher_is_better=ep.higher_is_better)
    return out


def aggregate_improvement(candidate: MetricsBundle, baseline: MetricsBundle) -> float:
    """Unweighted mean of I_t. Falls back to MAE if endpoints are empty."""
    vec = endpoint_improvements(candidate, baseline)
    if vec:
        return float(sum(vec.values()) / len(vec))
    return normalised_improvement(candidate.mae, baseline.mae, higher_is_better=False)


def robust_score(candidate: MetricsBundle, baseline: MetricsBundle) -> float:
    """Mixture of mean MAE gain, worst-airport MAE gain, and hazard CSI gain."""
    mae_gain = normalised_improvement(candidate.mae, baseline.mae, higher_is_better=False)
    csi_gain = normalised_improvement(candidate.hazard_csi, baseline.hazard_csi, higher_is_better=True)
    worst_cand = max(candidate.per_airport_mae.values()) if candidate.per_airport_mae else candidate.mae
    worst_base = max(baseline.per_airport_mae.values()) if baseline.per_airport_mae else baseline.mae
    worst_gain = normalised_improvement(worst_cand, worst_base, higher_is_better=False)
    return 0.5 * mae_gain + 0.3 * csi_gain + 0.2 * worst_gain


def safety_ok(candidate: MetricsBundle, baseline: MetricsBundle, tolerance: float) -> bool:
    return candidate.hazard_csi + 1e-12 >= baseline.hazard_csi - tolerance


def classify_signature(val_i: float, test_i: float, *, tau: float = 0.005) -> Signature:
    """Paper §5.1: paired val/test returns, two non-transfer signatures."""
    if val_i < tau and test_i < tau:
        return "none"
    if val_i > 0 and test_i < 0:
        return "distribution_shift"
    if val_i >= tau and test_i < tau:
        return "selection_variance"
    return "transfer"


def make_endpoints(
    *,
    per_airport_mae: dict[str, float],
    hazard_csi: float,
    overall_mae: float | None = None,
) -> dict[str, EndpointScore]:
    endpoints: dict[str, EndpointScore] = {}
    if overall_mae is not None:
        endpoints["overall.mae"] = EndpointScore(
            name="overall.mae",
            value=float(overall_mae),
            higher_is_better=False,
            kind="mae",
        )
    for airport, mae in sorted(per_airport_mae.items()):
        endpoints[f"{airport}.mae"] = EndpointScore(
            name=f"{airport}.mae",
            value=float(mae),
            higher_is_better=False,
            kind="mae",
        )
    endpoints["hazard.csi"] = EndpointScore(
        name="hazard.csi",
        value=float(hazard_csi),
        higher_is_better=True,
        kind="csi",
    )
    return endpoints
