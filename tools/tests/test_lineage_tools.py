import tempfile
import unittest
from pathlib import Path

from contract_tools.lineage_tools import (
    campaign_stats,
    load_records,
    merge_lineages,
    write_lineage,
)


def _done(tid, axis, status, verdict, cat="llm_template", gpu=2.5, at="2026-08-27T10:00:00Z", sp=False):
    return {"event": "trial_done", "trial_id": tid, "axis": axis, "status": status,
            "hypothesis_verdict": verdict, "arm_category": cat, "gpu_hours": gpu,
            "screen_pass": sp, "at": at}


class TestCampaignStats(unittest.TestCase):
    def test_counts_and_backfill_override(self):
        records = [
            _done("t1", "representation", "completed", "not_evaluated", at="2026-08-27T09:00:00Z"),
            {"event": "verdict_backfill", "trial_id": "t1", "hypothesis_verdict": "refuted"},
            _done("t2", "objective_tier1", "failed", "not_evaluated"),
            _done("t3", "objective_tier1", "completed", "supported", cat="llm_free", sp=True),
            {"event": "smoke_rejected", "trial_id": "t4", "axis": "representation"},
        ]
        s = campaign_stats(records)
        self.assertEqual(s["trials_done"], 3)
        self.assertEqual(s["verdicts"], {"refuted": 1, "not_evaluated": 1, "supported": 1})
        self.assertEqual(s["screen_passes"], ["t3"])
        self.assertEqual(s["by_arm_category"], {"llm_template": 2, "llm_free": 1})
        self.assertEqual(s["gpu_hours_total"], 7.5)
        self.assertEqual(s["events"]["smoke_rejected"], 1)


class TestMerge(unittest.TestCase):
    def _write(self, dirpath, name, records):
        p = Path(dirpath) / name
        write_lineage(records, p)
        return p

    def test_merge_dedup_and_sort(self):
        with tempfile.TemporaryDirectory() as td:
            shared = _done("t1", "representation", "completed", "refuted", at="2026-08-27T08:00:00Z")
            a = self._write(td, "a.jsonl", [shared, _done("t100", "representation", "completed", "refuted", at="2026-08-27T10:00:00Z")])
            b = self._write(td, "b.jsonl", [shared, _done("t200", "objective_tier1", "failed", "not_evaluated", at="2026-08-27T09:00:00Z")])
            merged = merge_lineages([a, b])
        self.assertEqual(len(merged), 3)  # shared 去重
        self.assertEqual([r["trial_id"] for r in merged], ["t1", "t200", "t100"])  # 按 at 排序

    def test_trial_id_conflict_raises(self):
        with tempfile.TemporaryDirectory() as td:
            a = self._write(td, "a.jsonl", [_done("t100", "representation", "completed", "refuted")])
            b = self._write(td, "b.jsonl", [_done("t100", "objective_tier1", "failed", "not_evaluated")])
            with self.assertRaises(ValueError):
                merge_lineages([a, b])

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jsonl"
            recs = [_done("t1", "representation", "completed", "refuted")]
            write_lineage(recs, p)
            self.assertEqual(load_records(p), recs)


if __name__ == "__main__":
    unittest.main()
