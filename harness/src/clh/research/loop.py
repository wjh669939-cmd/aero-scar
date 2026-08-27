"""Five-step closed-loop Auto Research driver."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from clh.config import HarnessConfig
from clh.core.context import HarnessContext
from clh.core.session import SessionLog
from clh.domain.protocol import load_adapter
from clh.llm.offline import OfflineLLM
from clh.llm.openai_compat import OpenAICompatLLM
from clh.llm.provider import LLMProvider
from clh.research.cards import TrialRecord
from clh.research.certification import certify_frozen, select_best_per_axis
from clh.research.evaluator import IndependentEvaluator
from clh.research.evidence import adjudicate
from clh.research.experiment import run_trial
from clh.research.specialist import Specialist


class ClosedLoopResearcher:
    """Hypothesis → action → experiment → independent eval → lineage, then certify."""

    def __init__(self, ctx: HarnessContext) -> None:
        self.ctx = ctx
        self.config = ctx.config
        self.session: SessionLog = ctx.get("session", SessionLog)
        self.llm: LLMProvider = ctx.get("llm")
        self.specialist = Specialist(self.llm, domain=self.config.domain.name)
        self.adapter = load_adapter(self.config, ctx.workspace_root)
        self.pristine = self.config.resolved_pipeline_root(ctx.workspace_root)
        if not (self.pristine / "pipeline.py").is_file():
            raise FileNotFoundError(f"pipeline_root missing pipeline.py: {self.pristine}")
        self.evaluator = IndependentEvaluator(self.config, self.adapter, self.pristine)
        self.trials: list[TrialRecord] = []
        self.trial_dirs: dict[str, str] = {}

    def run(self) -> dict[str, Any]:
        self.session.append("run_start", profile=self.config.profile, brief=self.adapter.describe())
        self.ctx.emit("research/start", run_dir=str(self.ctx.run_dir))
        for axis in self.config.research.axes:
            for index in range(self.config.research.budget_per_axis):
                self._one_trial(axis, index)
        best = select_best_per_axis(self.trials, self.config.research.selection_threshold)
        self.session.append("freeze", selected={axis: trial.trial_id for axis, trial in best.items()})
        cert = certify_frozen(self.evaluator, best, self.trial_dirs, self.config.certification)
        cert_path = self.ctx.run_dir / "certification.json"
        cert_path.write_text(json.dumps(cert, ensure_ascii=False, indent=2), encoding="utf-8")
        self.session.append("certification", path=str(cert_path), axes=list(best))
        summary = {
            "run_dir": str(self.ctx.run_dir),
            "n_trials": len(self.trials),
            "baseline_val": self.evaluator.baseline.to_dict(),
            "best": {axis: trial.model_dump() for axis, trial in best.items()},
            "certification": cert,
        }
        (self.ctx.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.ctx.emit("research/end", n_trials=len(self.trials))
        return summary

    def _one_trial(self, axis: str, index: int) -> TrialRecord:
        trial_id = f"{axis}-{index:03d}"
        self.ctx.emit("trial/start", trial_id=trial_id, axis=axis)
        hypothesis = self.specialist.hypothesize(axis, self.session.model_visible())  # type: ignore[arg-type]
        self.session.append("hypothesis", trial_id=trial_id, card=hypothesis.model_dump())
        action = self.specialist.act(hypothesis, self.pristine)
        self.session.append("action", trial_id=trial_id, card=action.model_dump(exclude={"files"}))
        trial_dir = self.ctx.run_dir / "trials" / trial_id
        try:
            metrics, changed = run_trial(
                pristine=self.pristine,
                trial_dir=trial_dir,
                action=action,
                evaluator=self.evaluator,
            )
            comparison = self.evaluator.compare(metrics)
            evidence = adjudicate(
                hypothesis,
                metrics,
                self.evaluator.baseline,
                self.config.research,
                ran_ok=True,
            )
            record = TrialRecord(
                trial_id=trial_id,
                axis=hypothesis.axis,
                hypothesis=hypothesis,
                action=ActionCardWithoutFiles(action),
                status=evidence.status,
                val_metrics=metrics,
                baseline_metrics=self.evaluator.baseline,
                improvement=float(comparison["improvement"]),
                evidence=evidence,
                notes=f"changed={changed}",
            )
        except Exception as exc:
            evidence = adjudicate(hypothesis, None, self.evaluator.baseline, self.config.research, ran_ok=False)
            record = TrialRecord(
                trial_id=trial_id,
                axis=hypothesis.axis,
                hypothesis=hypothesis,
                action=ActionCardWithoutFiles(action),
                status="failed",
                evidence=evidence,
                notes=str(exc),
            )
            if trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)
        else:
            self.trial_dirs[trial_id] = str(trial_dir)
        self.trials.append(record)
        self.session.append(
            "trial_result",
            trial_id=trial_id,
            status=record.status,
            improvement=record.improvement,
            evidence=evidence.model_dump(),
            val_metrics=record.val_metrics.to_dict() if record.val_metrics else None,
        )
        (self.ctx.run_dir / "trials" / f"{trial_id}.json").parent.mkdir(parents=True, exist_ok=True)
        (self.ctx.run_dir / "trials" / f"{trial_id}.json").write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )
        self.ctx.emit("trial/end", trial_id=trial_id, status=record.status)
        return record


def ActionCardWithoutFiles(action):  # noqa: N802
    from clh.research.cards import ActionCard

    payload = action.model_dump()
    payload["files"] = {name: f"<{len(body)} chars>" for name, body in action.files.items()}
    return ActionCard.model_validate(payload)


def build_llm(config: HarnessConfig) -> LLMProvider:
    if config.llm.provider == "offline" or not config.llm.api_key:
        return OfflineLLM()
    return OpenAICompatLLM(config.llm)
