"""随机对照臂驱动（零 LLM）：采样 → 机械生成编辑 → 与 LLM 臂完全相同的
闸门/训练/评测/裁决/记账链路。

用法（C1 评测器目录、source evaluator.env 的 shell）：
    python rand_arm_runner.py --n 6 --sample-seed 20260830 --start-seq 500

- 采样空间 random_space_v1（见 random_arm_edits.py 头注），均匀采模板后均匀采参数；
- trial 序号段 500-599（随机臂专属，防撞车）；
- 与 LLM 臂零特权差异：同 smoke/惰性闸门、同 axis_lock、同 v1.2 裁决、同复用与
  O2 数据通管；lineage 记 arm_category=random_arm。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/clh_deploy/discovery")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import discovery_runner as dr
from random_arm_edits import RANDOM_SPACE_V1, generate_edit, sample_params


def _params_key(action_id: str, params: dict) -> str:
    return json.dumps({"action_id": action_id, "params": params}, sort_keys=True, ensure_ascii=False)


def build_dedup_cache() -> dict[str, dict]:
    """扫描已有 rand-* trial，建 (模板,参数) → 已有结果 的缓存。

    重复采样时复用已记录结果、样本照常记账（不重排采样序列，
    均匀采样的统计性质不变）；位级复现性由 500/503 撞车对照背书。
    """
    cache: dict[str, dict] = {}
    trials_root = dr.DISCOVERY / "trials"
    if not trials_root.exists():
        return cache
    for tdir in sorted(trials_root.glob("rand-*")):
        tj, rj = tdir / "trial.json", tdir / "result.json"
        if not (tj.exists() and rj.exists()):
            continue
        trial = json.loads(tj.read_text(encoding="utf-8"))
        result = json.loads(rj.read_text(encoding="utf-8"))
        if result.get("status") == "completed":
            cache[_params_key(trial["action_id"], trial["params"])] = {
                "source_trial": trial["trial_id"], "result": result,
            }
    return cache


def run_random_trial(action: dict, params: dict, trial_seq: int, seed: int) -> dict:
    axis, new_source = generate_edit(action["action_id"], params)
    short = "rep" if axis == "representation" else "obj"
    trial_id = f"rand-{short}-{trial_seq:03d}"
    trial_dir = dr.DISCOVERY / "trials" / trial_id
    if trial_dir.exists() and any(trial_dir.iterdir()):
        rec = {"event": "driver_error", "axis": axis, "trial_seq": trial_seq,
               "error": f"trial_id {trial_id} 已被占用", "at": dr.now_utc()}
        dr.append_lineage(rec)
        return rec
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial = {
        "trial_id": trial_id,
        "axis": axis,
        "action_id": action["action_id"],
        "params": params,
        "hypothesis": action.get("hypothesis", ""),
        "falsification": action.get("falsification", ""),
        "is_free_proposal": False,
        "arm": "random",
        "screening_seed": seed,
    }
    (trial_dir / "trial.json").write_text(json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8")
    (trial_dir / "edit.py").write_text(new_source, encoding="utf-8")

    target = dr.AXIS_FILES[axis]
    current = target.read_text(encoding="utf-8")

    smoke_err = dr.functional_smoke(axis, trial_dir / "edit.py")
    if smoke_err:
        rec = {"event": "smoke_rejected", "trial_id": trial_id, "axis": axis,
               "action_id": action["action_id"], "arm_category": "random_arm",
               "smoke_error": smoke_err[:400], "at": dr.now_utc()}
        dr.append_lineage(rec)
        return rec

    lock_cfg = dr.load_config(dr.CONTRACT / "axis_lock_v1_draft.json", dr.AEROWF_REPO, dr.DOWNSTREAM)
    dec = dr.check([str(target)], axis, lock_cfg)
    if not dec.allowed:
        rec = {"event": "axis_lock_rejected", "trial_id": trial_id, "axis": axis,
               "violations": dec.violations, "at": dr.now_utc()}
        dr.append_lineage(rec)
        return rec

    backup = target.with_suffix(".py.discovery_backup")
    shutil.copy2(target, backup)
    started = time.monotonic()
    try:
        target.write_text(new_source, encoding="utf-8")
        cfg = dr.PipelineConfig(
            pipeline_script=Path(dr.DOWNSTREAM) / "src/aerowf_full_pipeline_v2.py",
            workdir=Path(dr.AEROWF_REPO),
            output_root_base=Path(dr.DOWNSTREAM) / "results/harness",
            python_bin="/root/miniconda3/bin/python",
            pretrain_epochs=100,
            downstream_epochs=30,
            timeout_sec=6 * 3600,
        )
        extra_args: list[str] = []
        if axis != dr.TIER2_AXIS and seed in dr.REUSE_PRETRAIN_CKPTS:
            extra_args = ["--reuse-pretrain-checkpoint", dr.REUSE_PRETRAIN_CKPTS[seed]]
        if axis == "objective_tier1":
            extra_args += ["--o2-event-flags", dr.O2_EVENT_FLAGS_CSV]
        outcome = dr.run_pipeline(cfg, seed=seed,
                                  run_id=f"disc_{trial_id.replace('-', '_')}_seed{seed}",
                                  extra_args=extra_args)
        result = {
            "trial_id": trial_id, "attempt_id": 1, "status": "failed",
            "status_reason": "; ".join(outcome.failure_reasons)[:500],
            "metrics_by_endpoint": {}, "paired_delta_vs_parent": {},
            "artifact_digests": {"edit_sha256": dr.sha256_text(new_source),
                                 **{f"ckpt.{k}": v for k, v in outcome.checkpoint_sha256.items()}},
            "resource_usage": {"gpu_hours": 0.0, "gpu_model": "RTX3090"},
            "hypothesis_verdict": "not_evaluated",
            "execution_deviations": [], "evaluation_manifest_digest": "",
            "created_at_utc": dr.now_utc(),
        }
        if outcome.status == "success":
            flat, deltas, combined = dr.eval_all_legs(
                outcome.output_root, trial_id, seed, trial_dir / "eval",
                parent_refs=dr.PARENT_REFS,
            )
            result["status"] = combined
            result["status_reason"] = "random-arm screening 单 seed 完成" if combined == "completed" else "评测未全部 completed"
            result["metrics_by_endpoint"] = {k: v for k, v in flat.items() if not k.startswith("__")}
            result["paired_delta_vs_parent"] = deltas
            result["evaluation_manifest_digest"] = flat.get("__manifest_digest__", "")
            result["guardrail_check"] = dr.guardrail_advisory(deltas)
            primary = deltas.get("forecast_scratch.RMSE_macro_norm")
            if primary is not None:
                result["screen_pass"] = bool(primary <= -0.0005)
        verdict, verdict_basis = dr.adjudicate(result, axis=axis)
        result["hypothesis_verdict"] = verdict
        tail = ""
        if outcome.status != "success":
            tail = dr.stderr_tail(cfg.output_root_base, outcome.run_id)
        result["resource_usage"]["gpu_hours"] = round((time.monotonic() - started) / 3600, 3)
        dr.validate_result({k: v for k, v in result.items() if k != "screen_pass"})
        (trial_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        dr.append_lineage({
            "event": "trial_done", "trial_id": trial_id, "axis": axis,
            "action_id": action["action_id"], "params": params,
            "parent_trial": "parent-seed42-formal",
            "arm_category": "random_arm",
            "hypothesis": trial["hypothesis"],
            "falsification": trial["falsification"], "status": result["status"],
            "status_reason": result["status_reason"],
            "stderr_tail": tail[:400],
            "paired_delta_vs_parent": result["paired_delta_vs_parent"],
            "screen_pass": result.get("screen_pass"),
            "hypothesis_verdict": verdict, "verdict_basis": verdict_basis,
            "guardrail_advisory": result.get("guardrail_check", {}).get("breaches"),
            "gpu_hours": result["resource_usage"]["gpu_hours"], "at": dr.now_utc(),
        })
        return result
    finally:
        shutil.copy2(backup, target)
        backup.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--sample-seed", type=int, required=True)
    ap.add_argument("--start-seq", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    registry = json.loads(dr.REGISTRY.read_text(encoding="utf-8"))
    actions = {a["action_id"]: a for a in registry["actions"]}
    rng = random.Random(args.sample_seed)
    dedup = build_dedup_cache()
    for i in range(args.n):
        seq = args.start_seq + i
        action_id = rng.choice(RANDOM_SPACE_V1)
        action = actions[action_id]
        params = sample_params(action, rng)
        print(f"\n===== RANDOM TRIAL {seq} action={action_id} params={params} {dr.now_utc()} =====", flush=True)
        key = _params_key(action_id, params)
        if key in dedup:
            hit = dedup[key]
            dr.append_lineage({
                "event": "trial_dedup_reuse", "trial_seq": seq, "axis": hit["result"].get("axis", ""),
                "action_id": action_id, "params": params,
                "arm_category": "random_arm",
                "reused_from": hit["source_trial"],
                "paired_delta_vs_parent": hit["result"].get("paired_delta_vs_parent"),
                "screen_pass": hit["result"].get("screen_pass"),
                "hypothesis_verdict": hit["result"].get("hypothesis_verdict"),
                "note": "重复 (模板,参数) 采样，复用已有结果记账，零 GPU",
                "at": dr.now_utc(),
            })
            print(f"TRIAL_OUTCOME: dedup reuse of {hit['source_trial']} (zero GPU)", flush=True)
            continue
        try:
            out = run_random_trial(action, params, seq, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"TRIAL_DRIVER_ERROR: {exc}", flush=True)
            dr.append_lineage({"event": "driver_error", "axis": "random_arm", "trial_seq": seq,
                               "error": str(exc)[:300], "at": dr.now_utc()})
            continue
        if out.get("status") == "completed":
            dedup[key] = {"source_trial": out["trial_id"], "result": out}
        print("TRIAL_OUTCOME:", json.dumps(
            {k: out.get(k) for k in ("trial_id", "event", "status", "paired_delta_vs_parent",
                                     "screen_pass", "hypothesis_verdict")},
            ensure_ascii=False), flush=True)
    print("\nBATCH_DONE", dr.now_utc(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
