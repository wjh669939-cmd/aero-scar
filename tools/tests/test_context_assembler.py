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
        "trial_id": "T-001",
        "axis": "representation",
        "action_id": "R1-wxcode-embedding",
        "verdict": "refuted",
        "primary_metric": {"name": "RMSE_macro_norm", "value": 0.0492},
        "verdict_note": "hazard CSI 无配对改善",
    },
    {
        "trial_id": "T-002",
        "axis": "representation",
        "action_id": "R3-runway-frame",
        "verdict": "supported",
        "primary_metric": {"name": "RMSE_macro_norm", "value": 0.0471},
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
                        "trial_id": "T-BAD",
                        "axis": "representation",
                        "verdict_note": "在 ZBAD 上也测过效果不错",
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
            {"trial_id": f"T-{i:03d}", "axis": "representation", "verdict": "refuted"}
            for i in range(30)
        ]
        text = summarize_lineage(many, max_records=10)
        self.assertNotIn("T-000", text)
        self.assertIn("T-029", text)
        self.assertEqual(len(text.splitlines()), 10)


if __name__ == "__main__":
    unittest.main()
