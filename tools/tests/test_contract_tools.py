"""contract_tools 单测：schema 校验、axis lock、随机臂采样。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_ROOT))

from contract_tools.axis_lock import LockDecision, check, load_config
from contract_tools.random_arm import load_registry, sample_trials, sampleable_actions
from contract_tools.validate import ContractViolation, validate_result, validate_trial

CONTRACT_DIR = TOOLS_ROOT.parent / "00_contract"


def _valid_trial() -> dict:
    return {
        "trial_id": "llm-rep-001",
        "arm": "llm",
        "axis": "representation",
        "tier": 1,
        "parent_trial": "parent-scratch-5seed",
        "action_id": "R4-runway-frame-wind",
        "hypothesis": "跑道坐标系风分量提供显式几何先验，改善侧风预测精度",
        "evidence_anchor": "官方指标含 Crosswind MAE 而输入为 u/v 分解",
        "target_slices": ["crosswind_mae@T+1/T+4/T+8"],
        "falsification": ">=2 机场 crosswind MAE 无配对改善",
        "editable_paths": ["src/features/**"],
        "budget": {"gpu_hours_cap": 1.0, "seeds": [42]},
        "created_at_utc": "2026-09-01T08:00:00Z",
    }


class TestValidate(unittest.TestCase):
    def test_valid_trial_passes(self):
        validate_trial(_valid_trial())

    def test_bad_trial_id_rejected(self):
        t = _valid_trial()
        t["trial_id"] = "my-cool-trial"
        with self.assertRaises(ContractViolation):
            validate_trial(t)

    def test_missing_falsification_rejected(self):
        t = _valid_trial()
        del t["falsification"]
        with self.assertRaises(ContractViolation):
            validate_trial(t)

    def test_model_axis_requires_param_fields(self):
        t = _valid_trial()
        t["trial_id"] = "llm-model-001"
        t["axis"] = "model"
        with self.assertRaises(ContractViolation) as ctx:
            validate_trial(t)
        self.assertIn("model_axis_extra", str(ctx.exception))

    def test_result_three_state_semantics(self):
        base = {
            "trial_id": "llm-rep-001",
            "attempt_id": 1,
            "evaluation_manifest_digest": "sha256:abc",
            "resource_usage": {"gpu_hours": 0.7, "gpu_model": "RTX3090"},
            "created_at_utc": "2026-09-01T09:00:00Z",
        }
        for status in ("completed", "invalid", "failed"):
            validate_result({**base, "status": status})
        with self.assertRaises(ContractViolation):
            validate_result({**base, "status": "success"})


class TestAxisLock(unittest.TestCase):
    """按 8/26 填实版 axis_lock：双根占位符 + 抽薄后真实路径。"""

    def setUp(self):
        self.config = load_config(
            CONTRACT_DIR / "axis_lock_v1_draft.json",
            repo_root="/repo/AeroWF",
            downstream_root="/repo/downstream",
        )

    def test_in_axis_edit_allowed(self):
        d = check(
            ["/repo/downstream/src/trial_features.py"], "representation", self.config
        )
        self.assertTrue(d.allowed)

    def test_objective_tier1_edit_allowed(self):
        d = check(
            ["/repo/downstream/src/trial_objective.py"], "objective_tier1", self.config
        )
        self.assertTrue(d.allowed)

    def test_cross_axis_edit_rejected(self):
        d = check(
            [
                "/repo/downstream/src/trial_features.py",
                "/repo/downstream/src/trial_objective.py",
            ],
            "representation",
            self.config,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.violations, ["/repo/downstream/src/trial_objective.py"])

    def test_training_loop_edit_rejected(self):
        d = check(
            ["/repo/downstream/src/aerowf_forecast_train_v2.py"],
            "objective_tier1",
            self.config,
        )
        self.assertFalse(d.allowed)

    def test_evaluator_edit_always_rejected(self):
        d = check(["/repo/evaluator/metrics.py"], "objective_tier1", self.config)
        self.assertFalse(d.allowed)

    def test_inactive_axis_rejected(self):
        d = check(
            ["/repo/AeroWF/models/AirFM/fusion/gate.py"], "model", self.config
        )
        self.assertFalse(d.allowed)
        self.assertIn("not active", d.reason)

    def test_empty_diff_rejected(self):
        d = check([], "representation", self.config)
        self.assertFalse(d.allowed)
        self.assertIn("empty diff", d.reason)

    def test_glob_star_pattern(self):
        config = dict(self.config)
        config["axes"] = {
            **config["axes"],
            "model": {**config["axes"]["model"], "active": True},
        }
        d = check(
            ["/repo/AeroWF/models/AirFM/fusion/deep/gate.py"], "model", config
        )
        self.assertTrue(d.allowed)

    def test_hidden_token_rejected(self):
        config = dict(self.config)
        config["forbidden_tokens"] = ["sealed_2026"]
        d = check(
            ["/repo/downstream/src/sealed_2026_peek.py"], "representation", config
        )
        self.assertFalse(d.allowed)
        self.assertIn("hidden-data", d.reason)

    def test_real_zbad_token_rejected(self):
        d = check(
            ["/data/ZBAD/runway.npy", "/repo/downstream/src/trial_features.py"],
            "representation",
            self.config,
        )
        self.assertFalse(d.allowed)
        self.assertIn("hidden-data", d.reason)


class TestRandomArm(unittest.TestCase):
    def test_sampled_trials_conform_to_schema(self):
        trials = sample_trials(n=8, seed=123, axis="representation")
        self.assertEqual(len(trials), 8)
        for t in trials:
            self.assertEqual(t["arm"], "random")
            self.assertEqual(t["axis"], "representation")

    def test_reproducible_given_seed(self):
        a = sample_trials(n=5, seed=99)
        b = sample_trials(n=5, seed=99)
        strip = lambda ts: [{k: v for k, v in t.items() if k != "created_at_utc"} for t in ts]
        self.assertEqual(strip(a), strip(b))

    def test_only_active_actions_sampled(self):
        registry = load_registry()
        pool_ids = {a["action_id"] for a in sampleable_actions(registry)}
        conditional = {
            a["action_id"] for a in registry["actions"] if a["status"] == "conditional"
        }
        self.assertFalse(pool_ids & conditional)
        trials = sample_trials(n=30, seed=7)
        for t in trials:
            self.assertIn(t["action_id"], pool_ids)

    def test_subset_params_nonempty(self):
        trials = sample_trials(n=40, seed=11, axis="representation")
        for t in trials:
            for name, value in t.get("sampled_params", {}).items():
                if isinstance(value, list):
                    self.assertGreater(len(value), 0, f"{t['action_id']}.{name} empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
