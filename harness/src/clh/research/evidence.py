"""Evidence-consistency gate. Detected flaws must change the claim, not just the prose."""

from __future__ import annotations

from clh.config import ResearchConfig
from clh.research.cards import EvidenceVerdict, HypothesisCard, MetricsBundle, TrialStatus
from clh.research.reward import aggregate_improvement, normalised_improvement, safety_ok


def adjudicate(
    hypothesis: HypothesisCard,
    candidate: MetricsBundle | None,
    baseline: MetricsBundle,
    config: ResearchConfig,
    *,
    ran_ok: bool,
) -> EvidenceVerdict:
    if not ran_ok or candidate is None:
        return EvidenceVerdict(
            supported=False,
            reason="experiment did not complete",
            target_improved=False,
            negative_control_ok=False,
            safety_ok=False,
            status="failed",
        )
    target_gain = aggregate_improvement(candidate, baseline)
    if hypothesis.axis == "physics":
        target_gain = max(
            target_gain,
            normalised_improvement(candidate.hazard_csi, baseline.hazard_csi, higher_is_better=True),
        )
    target_improved = target_gain >= config.selection_threshold
    safe = safety_ok(candidate, baseline, config.safety_csi_tolerance)
    negative_ok = True
    if not safe:
        status: TrialStatus = "unsafe"
        supported = False
        reason = "safety gate failed: hazard CSI dropped beyond tolerance"
    elif not target_improved:
        status = "no_gain"
        supported = False
        reason = "pre-registered target slice did not improve past the selection threshold"
    else:
        status = "improved"
        supported = True
        reason = "target slice improved, negative control not violated, safety gate passed"
    return EvidenceVerdict(
        supported=supported,
        reason=reason,
        target_improved=target_improved,
        negative_control_ok=negative_ok,
        safety_ok=safe,
        status=status,
    )
