"""生成本机 parent 参照（克隆机跨机不可位复现整改，2026-08-30）。

基线代码零编辑，走与 trial 完全相同的管线+评测路径；产出四腿指标
作为本机 paired delta 参照。"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/root/autodl-tmp/clh_deploy/discovery")
import discovery_runner as dr

cfg = dr.PipelineConfig(
    pipeline_script=Path(dr.DOWNSTREAM) / "src/aerowf_full_pipeline_v2.py",
    workdir=Path(dr.AEROWF_REPO),
    output_root_base=Path(dr.DOWNSTREAM) / "results/harness",
    python_bin="/root/miniconda3/bin/python",
    pretrain_epochs=100, downstream_epochs=30, timeout_sec=6 * 3600,
)
outcome = dr.run_pipeline(cfg, seed=42, run_id="parent_local_ot1_seed42",
    extra_args=["--reuse-pretrain-checkpoint", dr.REUSE_PRETRAIN_CKPTS[42]])
print("pipeline:", outcome.status, outcome.failure_reasons)
if outcome.status == "success":
    out = Path("/root/autodl-tmp/clh_deploy/discovery/parent_local_ot1")
    flat, deltas, combined = dr.eval_all_legs(outcome.output_root, "parent-local-ot1", 42,
                                              out / "eval", parent_refs=dr.PARENT_REFS)
    (out / "flat_metrics.json").write_text(json.dumps(flat, ensure_ascii=False, indent=2, default=str))
    (out / "delta_vs_D_parent.json").write_text(json.dumps(deltas, ensure_ascii=False, indent=2))
    print("combined:", combined)
    print("机器偏移（本机 parent vs D parent）:", json.dumps(deltas, ensure_ascii=False))
print("LOCAL_PARENT_DONE")
