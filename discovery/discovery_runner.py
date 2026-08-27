"""AeroWF-v1 正式 discovery 驱动（LLM 臂，筛选期）。

每个 trial 的闭环：
  1. context_assembler 组装提案提示词（隐藏 token 检查）→ DeepSeek 出提案 JSON；
  2. proposal_parser 严格解析 → trial 记录（过冻结 schema）；
  3. 第二段 LLM：当前轴文件全文 + 接口合同 + patch_plan → 新文件全文；
     py_compile + 接口签名存在性 + 隐藏 token + 禁 import 检查；
  4. axis_lock 校验 → 备份并安装编辑；
  5. 正式 full_pipeline（100 预训练 + 30 下游 epoch，seed 42，六条件判定）；
  6. 四条下游腿全部过适配器 + C evaluator（v1.0.2）；
  7. result.json 过冻结 schema；lineage.jsonl 追加（配对 Δ vs parent seed42）；
  8. 还原轴文件（finally 保证）。

失败的 trial 完整落盘并写入 lineage（假设/失败原因可被下一轮提案看到），不进入候选比较。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/clh_deploy/workspace/runs/aerowf-v1/tools")

from contract_tools.axis_lock import check, load_config
from contract_tools.context_assembler import (
    INTERFACE_CONTRACTS,
    assemble_proposal_prompt,
    assert_no_hidden_tokens,
)
from contract_tools.evaluator_client import run_evaluator
from contract_tools.pipeline_executor import PipelineConfig, run_pipeline
from contract_tools.predictions_adapter import adapt_stage_npz
from contract_tools.proposal_parser import parse_llm_proposal
from contract_tools.validate import validate_result

CONTRACT = Path("/root/autodl-tmp/clh_deploy/workspace/runs/aerowf-v1/00_contract")
REGISTRY = CONTRACT / "action_registry_v1_draft.json"
AEROWF_REPO = "/root/autodl-tmp/aerowf_baseline/AeroWF"
DOWNSTREAM = "/root/autodl-tmp/aerowf_downstream_v2"
DISCOVERY = Path("/root/autodl-tmp/clh_deploy/discovery")
LINEAGE = DISCOVERY / "lineage.jsonl"
COUNTS = {"ZBAA": 3961, "ZSPD": 3961, "ZSSS": 3961}
EVAL_CMD = ["/root/miniconda3/bin/python", "-m", "evaluator.aerowf_evaluator"]

AXIS_FILES = {
    "representation": Path(DOWNSTREAM) / "src/trial_features.py",
    "objective_tier1": Path(DOWNSTREAM) / "src/trial_objective.py",
}
REQUIRED_SIGNATURES = {
    "representation": ["def build_forecast_inputs(", "def build_classification_inputs(", "class AllowedContextEncoder"],
    "objective_tier1": ["def forecast_loss(", "def compute_class_weights(", "def classification_loss("],
}
FORBIDDEN_IMPORT_PAT = re.compile(
    r"^\s*(?:import|from)\s+(?:aerowf_|trial_features|trial_objective)", re.MULTILINE
)

# parent seed42 参考（C evaluator v1.0.2，2026-08-27 落盘于 discovery/parent_refs/）
PARENT_REFS = {
    "forecast_scratch": {"RMSE_macro_norm": 0.04847144659294747, "MAE_macro_norm": 0.0254475019647884},
    "forecast_pretrained": {"RMSE_macro_norm": 0.05090555, "MAE_macro_norm": 0.02848562},
    "classification_scratch": {"classification_macro_f1": 0.75035, "classification_csi_macro": 0.62679, "hazard_class_f1": 0.63309},
    "classification_pretrained": {"classification_macro_f1": 0.80365, "classification_csi_macro": 0.69388, "hazard_class_f1": 0.78857},
}

FAILURE_SLICES_SUMMARY = (
    "(a) 预训练对 forecast 是负迁移：3 个 seed（42/43/2027）上 scratch RMSE 一致优于 "
    "pretrained（seed42: 0.0485 vs 0.0509），配对 SNR 3.48，三 seed 同号；对 classification "
    "则是正迁移但幅度不稳定（macro_f1 增益 +0.077/+0.007 波动）。"
    "(b) 预测误差随时距单调恶化：seed43 scratch 分时距 RMSE T+1 0.0444 / T+4 0.0489 / T+8 0.0511，"
    "长时距（T+8, 120min）是最弱切片。"
    "(c) ZSSS 的 T+1 wind_x 出现回归头微溢出（>1 的预测值），提示低风速/边界值段拟合粗糙。"
    "(d) hazard 类 F1 跨 seed 波动极大（0.53~0.79，配对 SD 0.095），val 中 hazard 信号仅 ZBAA "
    "有支撑（support=94），ZSPD/ZSSS val 标签近单类。"
    "(e) 输入侧风用 (wind_x, wind_y) 归一化分量表示，低风速时方向信息被幅值淹没。"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def call_llm(api_key: str, messages: list[dict], temperature: float, max_tokens: int = 8000) -> str:
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def load_lineage() -> list[dict]:
    if not LINEAGE.exists():
        return []
    return [json.loads(line) for line in LINEAGE.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_lineage(record: dict) -> None:
    with LINEAGE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


_SMOKE_SNIPPETS = {
    "representation": """
import importlib.util, sys
import numpy as np, torch
spec = importlib.util.spec_from_file_location("trial_mod", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
runway = np.random.rand(96, 3, 2).astype(np.float32)
mask = np.array([True, True, False])
exo_cat = {"weather_code": 2, "sky_condition": 1, "has_gust": 0, "is_cavok": 1}
exo_cont = np.array([0.5, 0.3, 0.0], dtype=np.float32)
out = m.build_forecast_inputs(runway, mask, exo_cat, exo_cont, norm_stats=None)
assert out["x"].shape == (96, 3, 2), f"forecast x shape {out['x'].shape}"
assert out["node_mask"].dtype == torch.bool
out2 = m.build_classification_inputs(runway, mask, exo_cat, exo_cont)
assert out2["x"].shape == (96, 3, 2), f"cls x shape {out2['x'].shape}"
enc = m.AllowedContextEncoder(sky_known_max=5)
cat = {"sky_condition": torch.tensor([1, 2]), "has_gust": torch.tensor([0, 1]), "is_cavok": torch.tensor([1, 0])}
z = enc(cat, torch.rand(2, 3))
assert z.shape[0] == 2 and torch.isfinite(z).all()
print("SMOKE_OK")
""",
    "objective_tier1": """
import importlib.util, sys
import numpy as np, torch
spec = importlib.util.spec_from_file_location("trial_mod", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pred = torch.rand(4, 3, 3, 2); target = torch.rand(4, 3, 3, 2)
node_mask = torch.tensor([[1, 1, 0]] * 4, dtype=torch.bool)
loss = m.forecast_loss(pred, target, node_mask)
assert loss.dim() == 0 and torch.isfinite(loss), f"forecast_loss bad: {loss}"
pred.requires_grad_(True)
loss2 = m.forecast_loss(pred, target, node_mask); loss2.backward()
assert pred.grad is not None and torch.isfinite(pred.grad).all()
w = m.compute_class_weights(np.array([1000, 200, 50]))
assert w.shape == (3,) and torch.isfinite(w).all()
logits = torch.randn(8, 3, requires_grad=True)
label = torch.tensor([0, 1, 2, 0, 1, 2, 0, -100])
closs = m.classification_loss(logits, label, class_weights=w)
assert closs.dim() == 0 and torch.isfinite(closs); closs.backward()
print("SMOKE_OK")
""",
}


def functional_smoke(axis: str, source_path: Path) -> str | None:
    """CPU 假张量冒烟：签名可调用、形状/有限性/反传均正常。返回错误文本或 None。"""
    import subprocess as sp
    snippet = _SMOKE_SNIPPETS[axis]
    proc = sp.run(
        ["/root/miniconda3/bin/python", "-c", snippet, str(source_path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode == 0 and "SMOKE_OK" in proc.stdout:
        return None
    return (proc.stderr or proc.stdout).strip()[-600:]


def stderr_tail(output_root_base: Path, run_id: str, n: int = 12) -> str:
    p = output_root_base / f"{run_id}.stderr"
    if not p.exists():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def gen_code_edit(api_key: str, axis: str, trial: dict, current_source: str) -> tuple[str | None, list[str]]:
    """第二段 LLM：把 patch_plan 落成新文件全文。返回 (新全文 or None, 错误列表)。"""
    contract_text = INTERFACE_CONTRACTS.get(axis, "")
    prompt = f"""You are the code-editing module of a closed-loop auto-research system.
Your approved research proposal for this trial:
{json.dumps({k: trial[k] for k in ('hypothesis', 'patch_plan', 'expected_effect', 'falsification')}, ensure_ascii=False, indent=1)}

You must now implement the patch_plan by rewriting ONE file completely.

INTERFACE CONTRACT (must hold, violating = trial fails before training):
{contract_text}

Hard rules:
- Keep every public function/class signature listed in the contract unchanged.
- Do not import the locked training scripts or the other axis file.
- Keep tensor shapes and dtypes returned by each function identical to the current file.
- Pure-Python + numpy + torch only. No file I/O, no network, no subprocess.
- The change must implement exactly the approved patch_plan, nothing else.

Current file content:
```python
{current_source}
```

Answer with the complete new file content in ONE ```python code block and nothing else."""
    errors: list[str] = []
    for attempt in range(2):
        try:
            raw = call_llm(api_key, [{"role": "user", "content": prompt}], temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"codegen LLM call failed: {exc}")
            continue
        m = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
        if not m:
            errors.append("codegen output missing ```python block")
            continue
        source = m.group(1)
        try:
            assert_no_hidden_tokens(source)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"hidden token in codegen output: {exc}")
            continue
        missing = [sig for sig in REQUIRED_SIGNATURES[axis] if sig not in source]
        if missing:
            errors.append(f"missing required signatures: {missing}")
            continue
        if FORBIDDEN_IMPORT_PAT.search(source):
            errors.append("forbidden import of locked/other-axis module")
            continue
        tmp = Path("/tmp/_codegen_check.py")
        tmp.write_text(source, encoding="utf-8")
        try:
            py_compile.compile(str(tmp), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"py_compile failed: {exc}")
            continue
        return source, errors
    return None, errors


def eval_all_legs(output_root: Path, trial_id: str, seed: int, out_dir: Path) -> tuple[dict, dict, str]:
    """四条腿评测。返回 (metrics_by_endpoint 扁平, policy_deltas, 综合状态)。"""
    flat: dict = {}
    deltas: dict = {}
    statuses: list[str] = []
    manifest_digest = ""
    for stage in ("forecast_scratch", "forecast_pretrained", "classification_scratch", "classification_pretrained"):
        task = "forecast" if stage.startswith("forecast") else "classification"
        npz = output_root / stage / "validation_predictions.npz"
        sub = out_dir / stage
        sub.mkdir(parents=True, exist_ok=True)
        try:
            adapt_stage_npz(npz, task, sub / "predictions.npz", COUNTS, allow_fill=False)
        except Exception as exc:  # noqa: BLE001
            flat[f"{stage}.__adapter_error__"] = str(exc)
            statuses.append("failed")
            continue
        meta = {"trial_id": trial_id, "arm": "llm", "seed": seed, "task": task, "checkpoint_digest": stage}
        (sub / "trial_meta.json").write_text(json.dumps(meta))
        ev = run_evaluator(EVAL_CMD, sub / "predictions.npz", sub / "eval_out", trial_meta_path=sub / "trial_meta.json")
        statuses.append(ev.status)
        if ev.status != "completed":
            flat[f"{stage}.__evaluator_status__"] = f"{ev.status}: {ev.status_reason[:200]}"
            continue
        pol = ev.raw_metrics.get("decision_policy_metrics", {})
        for key, entry in pol.items():
            flat[f"{stage}.policy.{key}"] = entry
            ref = PARENT_REFS.get(stage, {}).get(key)
            if ref is not None:
                deltas[f"{stage}.{key}"] = round(entry["value"] - ref, 6)
        for name, ep in ev.metrics_by_endpoint.items():
            flat[f"{stage}.{name}"] = ep
        mpath = sub / "eval_out" / "evaluation_manifest.json"
        if mpath.exists():
            manifest_digest = hashlib.sha256(mpath.read_bytes()).hexdigest()
    if all(s == "completed" for s in statuses):
        combined = "completed"
    elif "failed" in statuses:
        combined = "failed"
    else:
        combined = "invalid"
    flat["__manifest_digest__"] = manifest_digest
    return flat, deltas, combined


def run_one_trial(api_key: str, axis: str, trial_seq: int, seed: int) -> dict:
    lineage = load_lineage()
    prompt = assemble_proposal_prompt(
        axis=axis,
        registry_path=REGISTRY,
        lineage_records=lineage,
        failure_slices_summary=FAILURE_SLICES_SUMMARY,
    )
    assert_no_hidden_tokens(prompt)

    trial = None
    proposal_errors: list[str] = []
    for attempt in range(3):
        raw = call_llm(api_key, [{"role": "user", "content": prompt}], temperature=0.7)
        parsed = parse_llm_proposal(raw, trial_seq=trial_seq, parent_trial="parent-seed42-formal", screening_seed=seed)
        if parsed.ok and parsed.trial_record["axis"] == axis:
            trial = parsed.trial_record
            break
        proposal_errors.extend(parsed.errors or [f"axis mismatch: {parsed.trial_record.get('axis')}"])
    if trial is None:
        rec = {"event": "proposal_rejected", "axis": axis, "trial_seq": trial_seq,
               "errors": proposal_errors[:6], "at": now_utc()}
        append_lineage(rec)
        return rec

    trial_dir = DISCOVERY / "trials" / trial["trial_id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial.json").write_text(json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8")

    target = AXIS_FILES[axis]
    current = target.read_text(encoding="utf-8")
    new_source, gen_errors = gen_code_edit(api_key, axis, trial, current)
    if new_source is None:
        rec = {"event": "codegen_rejected", "trial_id": trial["trial_id"], "axis": axis,
               "hypothesis": trial["hypothesis"], "errors": gen_errors[:6], "at": now_utc()}
        append_lineage(rec)
        return rec
    (trial_dir / "edit.py").write_text(new_source, encoding="utf-8")

    # CPU 冒烟闸门（G-15）：假张量调用全部接口，坏编辑不烧 GPU；一次修复机会
    smoke_err = functional_smoke(axis, trial_dir / "edit.py")
    if smoke_err:
        repair_prompt = (
            "Your previous file edit failed a functional smoke test before training.\n"
            f"Error:\n{smoke_err}\n\n"
            "Fix the bug. Keep the same approved patch_plan and all interface signatures. "
            "Answer with the complete corrected file in ONE ```python block and nothing else.\n\n"
            f"Your previous file content:\n```python\n{new_source}\n```"
        )
        try:
            raw = call_llm(api_key, [{"role": "user", "content": repair_prompt}], temperature=0.2)
            m = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
            if m:
                candidate = m.group(1)
                assert_no_hidden_tokens(candidate)
                if not FORBIDDEN_IMPORT_PAT.search(candidate) and all(
                    sig in candidate for sig in REQUIRED_SIGNATURES[axis]
                ):
                    (trial_dir / "edit.py").write_text(candidate, encoding="utf-8")
                    smoke_err = functional_smoke(axis, trial_dir / "edit.py")
                    if smoke_err is None:
                        new_source = candidate
        except Exception as exc:  # noqa: BLE001
            smoke_err = f"repair round failed: {exc}"
    if smoke_err:
        rec = {"event": "smoke_rejected", "trial_id": trial["trial_id"], "axis": axis,
               "hypothesis": trial["hypothesis"], "smoke_error": smoke_err[:400], "at": now_utc()}
        append_lineage(rec)
        return rec

    lock_cfg = load_config(CONTRACT / "axis_lock_v1_draft.json", AEROWF_REPO, DOWNSTREAM)
    dec = check([str(target)], axis, lock_cfg)
    if not dec.allowed:
        rec = {"event": "axis_lock_rejected", "trial_id": trial["trial_id"], "axis": axis,
               "violations": dec.violations, "at": now_utc()}
        append_lineage(rec)
        return rec

    backup = target.with_suffix(".py.discovery_backup")
    shutil.copy2(target, backup)
    started = time.monotonic()
    try:
        target.write_text(new_source, encoding="utf-8")
        cfg = PipelineConfig(
            pipeline_script=Path(DOWNSTREAM) / "src/aerowf_full_pipeline_v2.py",
            workdir=Path(AEROWF_REPO),
            output_root_base=Path(DOWNSTREAM) / "results/harness",
            python_bin="/root/miniconda3/bin/python",
            pretrain_epochs=100,
            downstream_epochs=30,
            timeout_sec=6 * 3600,
        )
        outcome = run_pipeline(cfg, seed=seed, run_id=f"disc_{trial['trial_id'].replace('-', '_')}_seed{seed}")

        result = {
            "trial_id": trial["trial_id"],
            "attempt_id": 1,
            "status": "failed",
            "status_reason": "; ".join(outcome.failure_reasons)[:500],
            "metrics_by_endpoint": {},
            "paired_delta_vs_parent": {},
            "artifact_digests": {"edit_sha256": sha256_text(new_source),
                                 **{f"ckpt.{k}": v for k, v in outcome.checkpoint_sha256.items()}},
            "resource_usage": {"gpu_hours": 0.0, "gpu_model": "RTX3090"},
            "hypothesis_verdict": "not_evaluated",
            "execution_deviations": [],
            "evaluation_manifest_digest": "",
            "created_at_utc": now_utc(),
        }
        if outcome.status == "success":
            flat, deltas, combined = eval_all_legs(outcome.output_root, trial["trial_id"], seed, trial_dir / "eval")
            result["status"] = combined
            result["status_reason"] = "screening 单 seed 完成" if combined == "completed" else "评测未全部 completed"
            result["metrics_by_endpoint"] = {k: v for k, v in flat.items() if not k.startswith("__")}
            result["paired_delta_vs_parent"] = deltas
            result["evaluation_manifest_digest"] = flat.get("__manifest_digest__", "")
            primary = deltas.get("forecast_scratch.RMSE_macro_norm")
            if primary is not None:
                # decision_policy v1.1 筛选线：配对改善 >= 0.0005
                result["screen_pass"] = bool(primary <= -0.0005)
        tail = ""
        if outcome.status != "success":
            tail = stderr_tail(cfg.output_root_base, outcome.run_id)
        result["resource_usage"]["gpu_hours"] = round((time.monotonic() - started) / 3600, 3)
        validate_result({k: v for k, v in result.items() if k != "screen_pass"})
        (trial_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        append_lineage({
            "event": "trial_done", "trial_id": trial["trial_id"], "axis": axis,
            "action_id": trial["action_id"], "hypothesis": trial["hypothesis"],
            "falsification": trial["falsification"],             "status": result["status"],
            "status_reason": result["status_reason"],
            "stderr_tail": tail[:400],
            "paired_delta_vs_parent": result["paired_delta_vs_parent"],
            "screen_pass": result.get("screen_pass"),
            "gpu_hours": result["resource_usage"]["gpu_hours"], "at": now_utc(),
        })
        return result
    finally:
        shutil.copy2(backup, target)
        backup.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--plan", required=True, help="逗号分隔的轴序列，如 representation,objective_tier1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-seq", type=int, default=0)
    args = ap.parse_args()

    DISCOVERY.mkdir(parents=True, exist_ok=True)
    # 冒烟闸门自检：基线文件必须通过，否则是闸门自身的假张量形状错了
    for axis, path in AXIS_FILES.items():
        err = functional_smoke(axis, path)
        if err:
            print(f"SMOKE_GATE_SELFCHECK_FAILED axis={axis}: {err}", flush=True)
            return 1
    print("smoke gate 自检通过（基线文件）", flush=True)
    axes = [a.strip() for a in args.plan.split(",") if a.strip()]
    for i, axis in enumerate(axes):
        seq = args.start_seq + i
        print(f"\n===== TRIAL {seq} axis={axis} {now_utc()} =====", flush=True)
        try:
            out = run_one_trial(args.api_key, axis, seq, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"TRIAL_DRIVER_ERROR: {exc}", flush=True)
            append_lineage({"event": "driver_error", "axis": axis, "trial_seq": seq,
                            "error": str(exc)[:300], "at": now_utc()})
            continue
        print("TRIAL_OUTCOME:", json.dumps(
            {k: out.get(k) for k in ("trial_id", "event", "status", "paired_delta_vs_parent", "screen_pass")},
            ensure_ascii=False), flush=True)
    print("\nBATCH_DONE", now_utc(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
