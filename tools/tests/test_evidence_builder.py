import json
import tempfile
import unittest
from pathlib import Path

from contract_tools.evidence_builder import (
    STATIC_CAMPAIGN_FACTS,
    build_failure_slices,
    classification_slice_table,
    forecast_slice_table,
)


def _fc_metrics(base: float) -> dict:
    eps = []
    for ap in ("ZBAA", "ZSPD"):
        for h, bump in (("T+1", 0.0), ("T+4", 0.004), ("T+8", 0.006)):
            for comp in ("wind_x", "wind_y"):
                eps.append({
                    "name": f"forecast.{ap}.{h}.{comp}.rmse",
                    "value": base + bump, "ci95": [0, 1], "degenerate": False,
                })
                eps.append({
                    "name": f"forecast.{ap}.{h}.{comp}.mae",
                    "value": base / 2, "ci95": [0, 1], "degenerate": False,
                })
    return {"endpoints": eps, "decision_policy_metrics": {}}


def _cls_metrics(hazard_f1: float, degenerate_zsps: bool) -> dict:
    eps = [
        {"name": "classification.ZBAA.f1_good", "value": 0.99, "ci95": [0, 1], "degenerate": False},
        {"name": "classification.ZBAA.f1_class_hazard", "value": hazard_f1, "ci95": [0, 1], "degenerate": False},
        {"name": "classification.ZSPD.f1_class_hazard", "value": 0.0, "ci95": [0, 1], "degenerate": degenerate_zsps},
    ]
    return {
        "endpoints": eps,
        "decision_policy_metrics": {
            "classification_macro_f1": {"value": 0.75},
            "hazard_class_f1": {"value": hazard_f1},
        },
    }


class TestEvidenceBuilder(unittest.TestCase):
    def test_forecast_table_ranks_worst_and_counts_negative_transfer(self):
        table = forecast_slice_table(_fc_metrics(0.044), _fc_metrics(0.050))
        self.assertIn("T+8", table.splitlines()[0])  # 最弱切片应是 T+8
        self.assertIn("T+1=0.0440", table)
        self.assertIn("8/12 个", table.replace("12/12", "8/12") if "12/12" in table else table)
        self.assertIn("负迁移逐点统计", table)

    def test_negative_transfer_count_exact(self):
        table = forecast_slice_table(_fc_metrics(0.044), _fc_metrics(0.050))
        self.assertIn("12/12 个", table)

    def test_classification_table_flags_degenerate(self):
        table = classification_slice_table(_cls_metrics(0.63, True), _cls_metrics(0.79, True))
        self.assertIn("degenerate", table)
        self.assertIn("ZSPD.f1_class_hazard", table)
        self.assertIn("hazard_class_f1: scratch 0.630 vs pretrained 0.790", table)

    def test_build_full_text_with_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, payload in (
                ("seed42_forecast_scratch", _fc_metrics(0.044)),
                ("seed42_forecast_pretrained", _fc_metrics(0.050)),
                ("seed42_cls_scratch", _cls_metrics(0.63, True)),
                ("seed42_cls_pretrained", _cls_metrics(0.79, True)),
            ):
                out = root / name / "out"
                out.mkdir(parents=True)
                (out / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            text = build_failure_slices(root)
        self.assertIn(STATIC_CAMPAIGN_FACTS[:20], text)
        self.assertIn("[auto] parent(seed42) 预测网格", text)
        self.assertIn("[auto] parent(seed42) 分类网格", text)
        self.assertLessEqual(len(text), 2600)

    def test_missing_dir_falls_back_to_static_only(self):
        text = build_failure_slices(Path("/nonexistent_dir_xyz"))
        self.assertEqual(text, STATIC_CAMPAIGN_FACTS[:2600])


if __name__ == "__main__":
    unittest.main()
