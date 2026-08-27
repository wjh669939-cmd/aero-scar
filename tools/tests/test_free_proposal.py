import unittest

from contract_tools.free_proposal import (
    CONSECUTIVE_REFUTED_THRESHOLD,
    forced_free_status,
    validate_free_proposal,
)
from contract_tools.proposal_parser import parse_llm_proposal


def _done(trial_id, axis, action_id, verdict):
    return {"event": "trial_done", "trial_id": trial_id, "axis": axis,
            "action_id": action_id, "hypothesis_verdict": verdict}


TEMPLATES = ["O1-x", "O2-y", "O3-z"]


class TestForcedFreeStatus(unittest.TestCase):
    def test_below_threshold_not_forced(self):
        lineage = [_done("t1", "objective_tier1", "O1-x", "refuted")]
        forced, reason = forced_free_status(lineage, "objective_tier1", TEMPLATES)
        self.assertFalse(forced)
        self.assertIn("阈值", reason)

    def test_exemption_when_untried_templates_remain(self):
        lineage = [
            _done("t1", "objective_tier1", "O1-x", "refuted"),
            _done("t2", "objective_tier1", "O1-x", "refuted"),
        ]
        forced, reason = forced_free_status(lineage, "objective_tier1", TEMPLATES)
        self.assertFalse(forced)
        self.assertIn("豁免", reason)

    def test_forced_when_exhausted_and_consecutive_refuted(self):
        lineage = [
            _done("t1", "objective_tier1", "O1-x", "refuted"),
            _done("t2", "objective_tier1", "O2-y", "not_evaluated"),
            _done("t3", "objective_tier1", "O3-z", "refuted"),
            _done("t4", "objective_tier1", "O1-x", "refuted"),
        ]
        forced, reason = forced_free_status(lineage, "objective_tier1", TEMPLATES)
        self.assertTrue(forced)
        self.assertIn("穷举", reason)

    def test_supported_resets_trailing_count(self):
        lineage = [
            _done("t1", "objective_tier1", "O1-x", "refuted"),
            _done("t2", "objective_tier1", "O2-y", "refuted"),
            _done("t3", "objective_tier1", "O3-z", "supported"),
        ]
        forced, _ = forced_free_status(lineage, "objective_tier1", TEMPLATES)
        self.assertFalse(forced)

    def test_other_axis_records_ignored(self):
        lineage = [
            _done("t1", "representation", "R2-a", "refuted"),
            _done("t2", "representation", "R5-b", "refuted"),
        ]
        forced, _ = forced_free_status(lineage, "objective_tier1", TEMPLATES)
        self.assertFalse(forced)

    def test_verdict_backfill_overrides(self):
        lineage = [
            _done("t1", "objective_tier1", "O1-x", "not_evaluated"),
            _done("t2", "objective_tier1", "O2-y", "refuted"),
            _done("t3", "objective_tier1", "O3-z", "refuted"),
            {"event": "verdict_backfill", "trial_id": "t1", "hypothesis_verdict": "refuted"},
        ]
        forced, reason = forced_free_status(lineage, "objective_tier1", TEMPLATES)
        self.assertTrue(forced, reason)
        self.assertEqual(CONSECUTIVE_REFUTED_THRESHOLD, 2)


class TestFreeProposalValidation(unittest.TestCase):
    _BASE = {
        "axis": "objective_tier1", "action_id": "free-quantile-head", "tier": 1,
        "hypothesis": "分位数回归头改善极端风速段的欠拟合问题，机制在于非对称损失",
        "evidence_anchor": "wind_x 高值段误差集中，均方误差对尾部欠敏感",
        "target_slices": ["forecast_T+8"],
        "falsification": "T+8 RMSE 配对无改善即证伪",
        "editable_paths": ["<DOWNSTREAM>/src/trial_objective.py"],
        "patch_plan": "改写 forecast_loss 为 pinball loss 组合",
    }

    def test_parser_rejects_free_without_non_expressibility(self):
        import json
        parsed = parse_llm_proposal(json.dumps(self._BASE), 0, "p")
        self.assertFalse(parsed.ok)
        self.assertIn("non_expressibility", parsed.errors[0])

    def test_parser_accepts_free_with_non_expressibility(self):
        import json
        payload = dict(self._BASE)
        payload["non_expressibility"] = "现有模板只覆盖 MSE 的加权变体，无法表达分位数损失这种非对称目标函数族"
        parsed = parse_llm_proposal(json.dumps(payload), 0, "p")
        self.assertTrue(parsed.ok, parsed.errors)
        self.assertTrue(parsed.trial_record["is_free_proposal"])
        self.assertIn("分位数", parsed.trial_record["non_expressibility"])

    def test_template_proposal_unaffected(self):
        errors = validate_free_proposal({"is_free_proposal": False})
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
