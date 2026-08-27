"""G-10 适配器单测：合成 D 阶段 npz 验证合同转换与缺失处理。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract_tools.predictions_adapter import (  # noqa: E402
    AdapterError,
    AdaptResult,
    adapt_stage_npz,
    full_manifest_ids,
)

COUNTS = {"ZBAA": 10, "ZSPD": 10, "ZSSS": 10}


def _make_stage_npz(path: Path, task: str, drop_tail: int = 0):
    per = 10 - drop_tail
    n = per * 3
    airport_id = np.repeat([0, 1, 2], per)
    source_index = np.tile(np.arange(per), 3)
    if task == "forecast":
        prediction = np.random.default_rng(0).random((n, 4, 3, 2)).astype(np.float32)
    else:
        prediction = np.random.default_rng(0).integers(0, 3, size=n).astype(np.int64)
    np.savez(
        path,
        prediction=prediction,
        airport_id=airport_id,
        source_index=source_index,
        target=prediction,
    )


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_manifest_ids_format_and_order(self):
        ids = full_manifest_ids(COUNTS)
        self.assertEqual(len(ids), 30)
        self.assertEqual(ids[0], "processed:ZBAA:val:00000000")
        self.assertEqual(ids[10], "processed:ZSPD:val:00000000")
        self.assertEqual(ids[-1], "processed:ZSSS:val:00000009")

    def test_full_coverage_forecast(self):
        src = self.dir / "s.npz"
        _make_stage_npz(src, "forecast")
        res = adapt_stage_npz(src, "forecast", self.dir / "o.npz", COUNTS)
        self.assertEqual(res.n_missing_filled, 0)
        with np.load(res.out_path, allow_pickle=False) as z:
            self.assertEqual(set(z.files), {"sample_id", "pred"})
            self.assertEqual(z["pred"].shape, (30, 4, 3, 2))
            self.assertEqual(str(z["pred"].dtype), "float32")

    def test_missing_rejected_by_default(self):
        src = self.dir / "s.npz"
        _make_stage_npz(src, "forecast", drop_tail=2)
        with self.assertRaises(AdapterError):
            adapt_stage_npz(src, "forecast", self.dir / "o.npz", COUNTS)

    def test_missing_filled_when_allowed(self):
        src = self.dir / "s.npz"
        _make_stage_npz(src, "forecast", drop_tail=2)
        res: AdaptResult = adapt_stage_npz(
            src, "forecast", self.dir / "o.npz", COUNTS, allow_fill=True
        )
        self.assertEqual(res.n_missing_filled, 6)
        self.assertIn("processed:ZBAA:val:00000009", res.missing_sample_ids)
        with np.load(res.out_path, allow_pickle=False) as z:
            self.assertFalse(np.isnan(z["pred"]).any())
            self.assertEqual(len(z["sample_id"]), 30)

    def test_virtual_slot_neutralized_real_untouched(self):
        src = self.dir / "s.npz"
        n = 30
        rng = np.random.default_rng(1)
        prediction = rng.random((n, 4, 3, 2)).astype(np.float32)
        node_mask = np.ones((n, 4), dtype=bool)
        node_mask[:, 3] = False  # 槽位 3 为虚拟
        prediction[:, 3] = 9.9  # 虚拟槽位放越界垃圾值
        prediction[0, 0, 0, 0] = 0.123  # 真实跑道哨兵值
        np.savez(
            src,
            prediction=prediction,
            airport_id=np.repeat([0, 1, 2], 10),
            source_index=np.tile(np.arange(10), 3),
            node_mask=node_mask,
        )
        res = adapt_stage_npz(src, "forecast", self.dir / "o.npz", COUNTS)
        with np.load(res.out_path, allow_pickle=False) as z:
            pred = z["pred"]
            self.assertTrue(((pred >= 0) & (pred <= 1)).all())  # 垃圾值已中和
            self.assertAlmostEqual(float(pred[0, 0, 0, 0]), 0.123, places=5)  # 真实值未动
            self.assertTrue((pred[:, 3] == 0.5).all())

    def test_classification_int_contract(self):
        src = self.dir / "s.npz"
        _make_stage_npz(src, "classification")
        res = adapt_stage_npz(src, "classification", self.dir / "o.npz", COUNTS)
        with np.load(res.out_path, allow_pickle=False) as z:
            self.assertEqual(z["pred"].shape, (30,))
            self.assertEqual(z["pred"].dtype.kind, "i")
            self.assertTrue(((z["pred"] >= 0) & (z["pred"] <= 2)).all())


if __name__ == "__main__":
    unittest.main()
