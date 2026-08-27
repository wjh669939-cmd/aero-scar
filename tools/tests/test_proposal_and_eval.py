"""proposal_parser 与 evaluator_client 单测（自 simple_ar/aerowf_v1 移植并适配）。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

TOOLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_ROOT))

from contract_tools.evaluator_client import run_evaluator
from contract_tools.proposal_parser import parse_llm_proposal
from contract_tools.validate import ContractViolation, check_param_budget

VALID_PAYLOAD = {
    "axis": "representation",
    "action_id": "R4-runway-frame-wind",
    "tier": 1,
    "hypothesis": "跑道坐标系风分量提供显式几何先验，改善侧风预测精度",
    "evidence_anchor": "官方指标含 Crosswind MAE 而输入为 u/v 分解",
    "target_slices": ["crosswind_mae@T+1/T+4/T+8"],
    "expected_effect": "crosswind MAE 配对改善",
    "falsification": ">=2 机场 crosswind MAE 无配对改善",
    "editable_paths": ["src/features/**"],
    "patch_plan": "在特征构建中加入按跑道磁向旋转的风分量。",
}


class TestProposalParser(unittest.TestCase):
    def test_valid_proposal_parses(self):
        raw = "some preamble\n" + json.dumps(VALID_PAYLOAD, ensure_ascii=False)
        out = parse_llm_proposal(raw, trial_seq=7, parent_trial="parent-scratch-5seed")
        self.assertTrue(out.ok, out.errors)
        self.assertEqual(out.trial_record["trial_id"], "llm-rep-007")
        self.assertFalse(out.trial_record["is_free_proposal"])

    def test_free_proposal_flagged(self):
        payload = {**VALID_PAYLOAD, "action_id": "free-wind-shear-feature"}
        out = parse_llm_proposal(json.dumps(payload, ensure_ascii=False), 1, "parent")
        self.assertTrue(out.ok, out.errors)
        self.assertTrue(out.trial_record["is_free_proposal"])

    def test_no_json_rejected(self):
        out = parse_llm_proposal("I think we should try a GRU.", 1, "parent")
        self.assertFalse(out.ok)

    def test_missing_scientific_field_rejected(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "falsification"}
        out = parse_llm_proposal(json.dumps(payload, ensure_ascii=False), 1, "parent")
        self.assertFalse(out.ok)
        self.assertIn("falsification", str(out.errors))

    def test_model_axis_requires_extra(self):
        payload = {**VALID_PAYLOAD, "axis": "model"}
        out = parse_llm_proposal(json.dumps(payload, ensure_ascii=False), 1, "parent")
        self.assertFalse(out.ok)
        self.assertIn("model_axis_extra", str(out.errors))


class TestParamBudget(unittest.TestCase):
    def test_within_budget_passes(self):
        check_param_budget(4_200_000, 4_300_000)

    def test_over_budget_rejected(self):
        with self.assertRaises(ContractViolation):
            check_param_budget(4_200_000, 5_000_000)


class _FakeEvaluator:
    """生成一个把固定 JSON 写入 out-dir 的假 evaluator 脚本。"""

    def __init__(self, tmp: Path, payload: dict | None, exit_code: int = 0):
        self.script = tmp / "fake_eval.py"
        body = f"""
import json, sys
args = sys.argv
out_dir = args[args.index('--out-dir') + 1]
payload = {payload!r}
if payload is not None:
    with open(out_dir + '/metrics.json', 'w') as fh:
        json.dump(payload, fh)
sys.exit({exit_code})
"""
        self.script.write_text(body, encoding="utf-8")

    @property
    def cmd(self) -> list[str]:
        return [sys.executable, str(self.script)]


class TestEvaluatorClient(unittest.TestCase):
    def _run(self, payload, exit_code=0):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake = _FakeEvaluator(tmp_path, payload, exit_code)
            preds = tmp_path / "predictions.npz"
            preds.write_bytes(b"fake")
            return run_evaluator(fake.cmd, preds, tmp_path / "out")

    COMPLETED = {
        "status": "completed",
        "task": "classification",
        "endpoints": [
            {"name": "classification.ZBAA.macro_f1", "value": 0.68, "ci95": [0.45, 0.81], "degenerate": False},
            {"name": "classification.ZBAA.csi_macro", "value": 0.57, "ci95": [0.40, 0.70], "degenerate": False},
        ],
        "overall": [
            {"name": "overall.classification.macro_f1", "value": 0.68, "ci95": [0.45, 0.81], "degenerate": False},
        ],
        "anomaly_counts": {"nan": 0, "zero_support": 4},
    }

    def test_completed(self):
        out = self._run(self.COMPLETED)
        self.assertEqual(out.status, "completed")
        self.assertIn("classification.ZBAA.macro_f1", out.metrics_by_endpoint)
        self.assertEqual(out.metrics_by_endpoint["classification.ZBAA.macro_f1"]["value"], 0.68)
        self.assertIn("overall.classification.macro_f1", out.overall)
        self.assertEqual(out.anomaly_counts["zero_support"], 4)

    def test_invalid_passthrough_exit2(self):
        # C 冻结版：invalid 时 exit 2 + metrics.json 落盘（含 reason）
        out = self._run(
            {"status": "invalid", "not_evaluated": True, "reason": "sample_id 与完整 val 清单不一致",
             "anomaly_counts": {"missing_id": 24}},
            exit_code=2,
        )
        self.assertEqual(out.status, "invalid")
        self.assertIn("sample_id", out.status_reason)
        self.assertEqual(out.anomaly_counts["missing_id"], 24)

    def test_failed_passthrough_exit3(self):
        out = self._run(
            {"status": "failed", "not_evaluated": True, "reason": "未安装 C 私有配置"},
            exit_code=3,
        )
        self.assertEqual(out.status, "failed")
        self.assertIn("私有配置", out.status_reason)

    def test_completed_with_nonzero_exit_is_failed(self):
        # metrics 说 completed 但退出码非 0：交叉校验失败
        out = self._run(self.COMPLETED, exit_code=3)
        self.assertEqual(out.status, "failed")

    def test_missing_metrics_file_is_failed(self):
        out = self._run(None)
        self.assertEqual(out.status, "failed")
        self.assertIn("no metrics.json", out.status_reason)

    def test_completed_without_endpoints_is_failed(self):
        out = self._run({"status": "completed", "endpoints": []})
        self.assertEqual(out.status, "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
