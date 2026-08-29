"""G-5 上下文组装器单测：真实 registry 组装 + 隐藏 token 拦截。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract_tools.context_assembler import (  # noqa: E402
    HiddenInfoLeak,
    assemble_proposal_prompt,
    assert_no_hidden_tokens,
    summarize_lineage,
)

REGISTRY = (
    Path(__file__).resolve().parents[2] / "00_contract" / "action_registry_v1_draft.json"
)

LINEAGE = [
    {
        "event": "trial_done",
        "trial_id": "T-001",
        "axis": "representation",
        "action_id": "R1-wxcode-embedding",
        "arm_category": "llm_template",
        "status": "completed",
        "hypothesis_verdict": "refuted",
        "verdict_basis": "verdict_rule_v1: 主指标配对 Δ +0.0000 未达筛选线",
        "paired_delta_vs_parent": {"forecast_scratch.RMSE_macro_norm": 0.0},
        "hypothesis": "weather grouping as extra input feature",
    },
    {
        "event": "trial_done",
        "trial_id": "T-002",
        "axis": "representation",
        "action_id": "R3-runway-frame",
        "arm_category": "llm_template",
        "status": "completed",
        "hypothesis_verdict": "supported",
        "verdict_basis": "verdict_rule_v1: 主指标配对改善 0.0021 >= 筛选线",
        "paired_delta_vs_parent": {"forecast_scratch.RMSE_macro_norm": -0.0021},
    },
]


class ContextAssemblerTest(unittest.TestCase):
    def test_assemble_representation_prompt(self):
        prompt = assemble_proposal_prompt(
            axis="representation",
            registry_path=REGISTRY,
            lineage_records=LINEAGE,
            failure_slices_summary="- T+8 RMSE 恶化集中于阵风时段（val）",
        )
        self.assertIn("trial_features.py", prompt)
        self.assertIn("R1-wxcode-embedding", prompt)
        self.assertIn("T-002", prompt)
        self.assertIn('"axis": "representation"', prompt)
        self.assertNotIn("{AXIS}", prompt)
        self.assertNotIn("{LINEAGE_SUMMARY}", prompt)

    def test_objective_prompt_has_loss_contract(self):
        prompt = assemble_proposal_prompt(
            axis="objective_tier1",
            registry_path=REGISTRY,
            lineage_records=[],
            failure_slices_summary="- (none yet)",
        )
        self.assertIn("trial_objective.py", prompt)
        self.assertIn("(no prior trials in this campaign)", prompt)

    def test_unknown_axis_rejected(self):
        with self.assertRaises(ValueError):
            assemble_proposal_prompt(
                axis="physics",
                registry_path=REGISTRY,
                lineage_records=[],
                failure_slices_summary="",
            )

    def test_hidden_token_blocks_assembly(self):
        with self.assertRaises(HiddenInfoLeak):
            assemble_proposal_prompt(
                axis="representation",
                registry_path=REGISTRY,
                lineage_records=[
                    {
                        "event": "trial_done",
                        "trial_id": "T-BAD",
                        "axis": "representation",
                        "status": "completed",
                        "hypothesis_verdict": "refuted",
                        "verdict_basis": "在 ZBAD 上也测过效果不错",
                        "paired_delta_vs_parent": {},
                    }
                ],
                failure_slices_summary="- ok",
            )

    def test_assert_no_hidden_tokens_direct(self):
        assert_no_hidden_tokens("clean text about ZBAA and ZSPD")
        for bad in ("sealed/2026", "AeroWF_v1_EVALUATION", "test_metrics: 0.1"):
            with self.assertRaises(HiddenInfoLeak):
                assert_no_hidden_tokens(bad)

    def test_lineage_truncation(self):
        many = [
            {"event": "trial_done", "trial_id": f"T-{i:03d}", "axis": "representation",
             "status": "completed", "hypothesis_verdict": "refuted",
             "paired_delta_vs_parent": {}}
            for i in range(30)
        ]
        text = summarize_lineage(many, max_records=10)
        self.assertNotIn("T-000", text)
        self.assertIn("T-029", text)
        # 每条 trial_done 渲染两行（头行 + Δ 行；无 basis/hypothesis 时）
        self.assertEqual(len(text.splitlines()), 20)

    def test_lineage_renders_deltas_and_audit_notes(self):
        recs = [
            {"event": "trial_done", "trial_id": "T-X", "axis": "objective_tier1",
             "status": "completed", "hypothesis_verdict": "refuted",
             "paired_delta_vs_parent": {
                 "forecast_scratch.RMSE_macro_norm": -0.000451,
                 "classification_pretrained.hazard_class_f1": 0.032859,
             }},
            {"event": "audit_note", "trial_id": "T-X",
             "note": "edit weighted the slot axis, claimed horizon weighting untested"},
            {"event": "smoke_rejected", "trial_id": "T-Y", "axis": "objective_tier1",
             "smoke_error": "RuntimeError: expanded size mismatch"},
        ]
        text = summarize_lineage(recs)
        self.assertIn("fc_scratch RMSE -0.000451", text)
        self.assertIn("hazardF1 +0.032859", text)
        self.assertIn("AUDIT NOTE trial=T-X", text)
        self.assertIn("gate-rejected (smoke_rejected)", text)


if __name__ == "__main__":
    unittest.main()
