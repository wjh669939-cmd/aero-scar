"""G-7 执行器单测：用假 pipeline 脚本本地验证六条件判定与失败线。"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract_tools.pipeline_executor import (  # noqa: E402
    STAGES,
    PipelineConfig,
    judge_output_tree,
    make_run_id,
    run_pipeline,
)

FAKE_PIPELINE = textwrap.dedent(
    """
    import argparse, json, os, sys
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--pretrain-epochs", type=int)
    p.add_argument("--downstream-epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--min-delta", type=float)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--output-root", type=Path)
    args = p.parse_args()

    mode = os.environ.get("FAKE_MODE", "ok")
    stages = ["pretrain", "forecast_scratch", "forecast_pretrained",
              "classification_scratch", "classification_pretrained"]
    root = args.output_root

    if mode == "crash":
        sys.exit(3)

    for st in stages:
        d = root / st
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        metrics = {"status": "success", "test_used": False, "val_rmse": 0.05}
        if st.endswith("pretrained"):
            metrics["checkpoint_load"] = {"missing_keys": [], "unexpected_keys": []}
        if mode == "bad_status" and st == "forecast_scratch":
            metrics["status"] = "error"
        if mode == "test_used" and st == "pretrain":
            metrics["test_used"] = True
        if mode == "nan" and st == "classification_scratch":
            metrics["val_rmse"] = float("nan")
        if mode == "bad_keys" and st == "forecast_pretrained":
            metrics["checkpoint_load"] = {"missing_keys": ["core.w"], "unexpected_keys": []}
        (d / "metrics.json").write_text(json.dumps(metrics))
        if not (mode == "no_ckpt" and st == "classification_pretrained"):
            (d / "checkpoints" / "best_model.pth").write_bytes(b"weights-" + st.encode())

    if mode != "no_summary":
        (root / "pipeline_summary.json").write_text(json.dumps({"seed": args.seed}))
    """
)


class PipelineExecutorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        base = Path(self._tmp.name)
        script = base / "fake_pipeline.py"
        script.write_text(FAKE_PIPELINE)
        self.config = PipelineConfig(
            pipeline_script=script,
            workdir=base,
            output_root_base=base / "harness",
            python_bin=sys.executable,
            timeout_sec=60,
        )

    def tearDown(self):
        self._tmp.cleanup()
        os.environ.pop("FAKE_MODE", None)

    def _run(self, mode: str, **kw):
        os.environ["FAKE_MODE"] = mode
        return run_pipeline(self.config, seed=1001, **kw)

    def test_success_all_six_conditions(self):
        out = self._run("ok")
        self.assertEqual(out.status, "success", out.failure_reasons)
        self.assertEqual(set(out.checkpoint_sha256), set(STAGES))
        self.assertEqual(out.pipeline_summary["seed"], 1001)
        self.assertEqual(len(out.stage_metrics), 5)

    def test_nonzero_exit_fails(self):
        out = self._run("crash")
        self.assertEqual(out.status, "failed")
        self.assertTrue(any("非零退出码" in r for r in out.failure_reasons))

    def test_missing_summary_fails(self):
        out = self._run("no_summary")
        self.assertEqual(out.status, "failed")
        self.assertTrue(any("pipeline_summary" in r for r in out.failure_reasons))

    def test_bad_stage_status_fails(self):
        out = self._run("bad_status")
        self.assertEqual(out.status, "failed")
        self.assertTrue(any("forecast_scratch" in r and "success" in r for r in out.failure_reasons))

    def test_test_used_true_fails(self):
        out = self._run("test_used")
        self.assertEqual(out.status, "failed")
        self.assertTrue(any("test_used" in r for r in out.failure_reasons))

    def test_nan_metric_fails(self):
        out = self._run("nan")
        self.assertEqual(out.status, "failed")
        self.assertTrue(any("NaN/Inf" in r for r in out.failure_reasons))

    def test_missing_checkpoint_fails(self):
        out = self._run("no_ckpt")
        self.assertEqual(out.status, "failed")
        self.assertTrue(any("best_model.pth" in r for r in out.failure_reasons))

    def test_nonempty_keys_fails(self):
        out = self._run("bad_keys")
        self.assertEqual(out.status, "failed")
        self.assertTrue(any("keys 非空" in r for r in out.failure_reasons))

    def test_output_dir_conflict_refused_without_running(self):
        rid = make_run_id("trial", 1001)
        conflict = self.config.output_root_base / rid
        conflict.mkdir(parents=True)
        (conflict / "old.txt").write_text("existing experiment")
        out = self._run("ok", run_id=rid)
        self.assertEqual(out.status, "failed")
        self.assertTrue(any("非空" in r for r in out.failure_reasons))
        self.assertEqual((conflict / "old.txt").read_text(), "existing experiment")

    def test_judge_is_reusable_offline(self):
        out = self._run("ok")
        reasons, stage_metrics, summary = judge_output_tree(out.output_root, exit_code=0)
        self.assertEqual(reasons, [])
        self.assertEqual(len(stage_metrics), 5)
        self.assertEqual(summary["seed"], 1001)

    def test_failure_scene_preserved(self):
        out = self._run("bad_status")
        stdout = self.config.output_root_base / f"{out.run_id}.stdout"
        self.assertTrue(stdout.exists())
        self.assertTrue((out.output_root / "forecast_scratch" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
